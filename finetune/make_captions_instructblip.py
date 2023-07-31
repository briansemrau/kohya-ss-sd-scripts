import argparse
import glob
import os
import json
import random
import sys

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
sys.path.append(os.path.dirname(__file__))
import library.train_util as train_util

from transformers import Blip2ForConditionalGeneration, AutoProcessor, InstructBlipForConditionalGeneration
from peft import AutoPeftModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_transform(image):
    #return image
    scale = 224 / min(image.size)
    return transforms.Compose(
        [
            transforms.Resize((int(image.size[1] * scale), int(image.size[0] * scale)), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            #transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
        ]
    )(image)

# 共通化したいが微妙に処理が異なる……
class ImageLoadingTransformDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            # convert to tensor temporarily so dataloader will accept it
            tensor = image_transform(image)
        except Exception as e:
            print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (tensor, img_path)


def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch

@torch.no_grad()
def main(args):
    # fix the seed for reproducibility
    seed = args.seed  # + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.exists("blip"):
        args.train_data_dir = os.path.abspath(args.train_data_dir)  # convert to absolute path

        cwd = os.getcwd()
        print("Current Working Directory is: ", cwd)
        os.chdir("finetune")

    print(f"load images from {args.train_data_dir}")
    train_data_dir_path = Path(args.train_data_dir)
    image_paths = train_util.glob_images_pathlib(train_data_dir_path, args.recursive)
    print(f"found {len(image_paths)} images.")
    
    processor = AutoProcessor.from_pretrained(args.base_model)#, pad_token="</s>")
    print(processor.tokenizer.vocab_size, processor.tokenizer.pad_token_id, processor.tokenizer.eos_token_id, processor.tokenizer.bos_token_id, processor.tokenizer.unk_token_id)
    if args.peft_weights is not None:
        print(f"loading BLIP2 caption with LoRA: {args.peft_weights}")
        if args.peft_weights.startswith("./"):
            args.peft_weights = os.path.abspath(args.peft_weights) + "/"
        if args.use_8bit:
            model = AutoPeftModel.from_pretrained(args.peft_weights, load_in_8bit=True, torch_dtype=torch.float16)
        else:
            model = AutoPeftModel.from_pretrained(args.peft_weights, torch_dtype=torch.float16, device_map=DEVICE)
    else:
        print(f"loading BLIP2 caption: {args.caption_weights}")
        #model = Blip2ForConditionalGeneration.from_pretrained(args.caption_weights)
        if args.use_8bit:
            model = InstructBlipForConditionalGeneration.from_pretrained(args.caption_weights, load_in_8bit=True, torch_dtype=torch.float16)
        else:
            model = InstructBlipForConditionalGeneration.from_pretrained(args.caption_weights, torch_dtype=torch.float16, device_map=DEVICE)
    model.eval()
    print(model.language_model.lm_head.weight.shape)
    print("BLIP2 loaded")

    prompt = args.prompt
    prompt_inputs = processor.__call__(text=[prompt], return_tensors="pt").to(DEVICE)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # captioningする
    def run_batch(path_imgs):
        #inputs = processor(images=[im for _, im in path_imgs], return_tensors="pt").to(DEVICE, model.dtype)
        #imgs = inputs.pixel_values
        
        imgs = torch.stack([im for _, im in path_imgs]).to(DEVICE, dtype=model.dtype)

        #plt.imshow(np.transpose(imgs[0].clone().cpu().numpy().clip(0, 1), (1, 2, 0)))
        #plt.savefig("test.png")

        for k, v in prompt_inputs.items():
            if type(v) == torch.Tensor:
                prompt_inputs[k] = torch.stack([v[0]] * len(path_imgs))
        
        ids = model.generate(
            imgs, **prompt_inputs, top_p=args.top_p, max_length=args.max_length, min_length=args.min_length,
            suppress_tokens=[32000], pad_token_id=0,
        )
        #ids[(ids == -1).cumsum(dim=-1) >= 1] = 2
        captions = processor.batch_decode(ids, skip_special_tokens=True)

        for (image_path, _), caption in zip(path_imgs, captions):
            with open(os.path.splitext(image_path)[0] + args.caption_extension, "wt", encoding="utf-8") as f:
                f.write(caption + "\n")
                if args.debug:
                    print(image_path, caption)

    # 読み込みの高速化のためにDataLoaderを使うオプション
    if args.max_data_loader_n_workers is not None:
        dataset = ImageLoadingTransformDataset(image_paths)
        data = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.max_data_loader_n_workers,
            collate_fn=collate_fn_remove_corrupted,
            drop_last=False,
        )
    else:
        data = [[(None, ip)] for ip in image_paths]

    b_imgs = []
    for data_entry in tqdm(data, smoothing=0.0):
        for data in data_entry:
            if data is None:
                continue

            img_tensor, image_path = data
            if img_tensor is None:
                try:
                    raw_image = Image.open(image_path)
                    if raw_image.mode != "RGB":
                        raw_image = raw_image.convert("RGB")
                    img_tensor = image_transform(raw_image)
                except Exception as e:
                    print(f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}")
                    continue
            
            import matplotlib.pyplot as plt
            #plt.imshow(np.transpose(img_tensor.clone().cpu().numpy(), (1, 2, 0)))
            #plt.savefig("test.png")

            b_imgs.append((image_path, img_tensor))
            if len(b_imgs) >= args.batch_size:
                run_batch(b_imgs)
                b_imgs.clear()
    if len(b_imgs) > 0:
        run_batch(b_imgs)

    print("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument(
        "--caption_weights",
        type=str,
        default="Salesforce/instructblip-vicuna-7b",
        help="InstructBLIP caption weights / InstructBLIPのキャプション生成のための重み",
    )
    parser.add_argument(
        "--peft_weights",
        type=str,
        help="InstructBLIP caption weights / InstructBLIPのキャプション生成のための重み",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Salesforce/instructblip-vicuna-7b",
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="use 8bit model for InstructBLIP caption / InstructBLIP captionの8bitモデルを使う",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Briefly list every detail about the image.",
        help="prompt for instructblip captioning / instructblipのキャプション生成のためのプロンプト",
    )
    parser.add_argument(
        "--caption_extention",
        type=str,
        default=None,
        help="extension of caption file (for backward compatibility) / 出力されるキャプションファイルの拡張子（スペルミスしていたのを残してあります）",
    )
    parser.add_argument("--caption_extension", type=str, default=".caption", help="extension of caption file / 出力されるキャプションファイルの拡張子")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=None,
        help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）",
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p in Nucleus sampling / Nucleus sampling時のtop_p")
    parser.add_argument("--max_length", type=int, default=75, help="max length of caption / captionの最大長")
    parser.add_argument("--min_length", type=int, default=5, help="min length of caption / captionの最小長")
    parser.add_argument("--seed", default=42, type=int, help="seed for reproducibility / 再現性を確保するための乱数seed")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--recursive", action="store_true", help="search for images in subfolders recursively / サブフォルダを再帰的に検索する")

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    # スペルミスしていたオプションを復元する
    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention

    main(args)
