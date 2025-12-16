# Dataset

## Dataset for CLIP

### Caltech101

- Create a folder named `caltech-101/` under `$DATA`.
- Download `101_ObjectCategories.tar.gz` from http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz and extract the file under `$DATA/caltech-101`.
- Download `split_zhou_Caltech101.json` from this [link](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing) and put it under `$DATA/caltech-101`.

The directory structure should look like

```
$DATA/
|-- caltech-101/
|   |-- 101_ObjectCategories/
|   | split_zhou_Caltech101.json
```

### mini-imagenet

- Create a folder named `mini-imagenet/` under `$DATA`.
- Download the dataset from the [mini-imagnet](https://yaoyaoliu.web.illinois.edu/projects/mtl/download/) and extract the training and validation sets to `$DATA/mini-imagenet`.
- Download the `classnames.txt` to `$DATA/mini-imagenet/` from this [link](https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view?usp=sharing). The class names are copied from [CLIP](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb).

The directory structure should look like

```
$DATA/
|–– mini-imagenet/
|   |–– train/ # contains 1,000 folders like n01440764, n01443537, etc.
|   |–– val/
|   |–– test/
|   |-- classnames.txt
```

### MSCOCO2014

- Create a folder named `mscoco2014/` under `$DATA`.
- Download the dataset from the [MSCOCO](https://cocodataset.org/#download) and extract the training and validation sets to `$DATA/mscoco2014`.
- download json file from `https://www.kaggle.com/datasets/wangjilong/dataset-json/` data/mscococ2014/\*.json to `$DATA/mscoco2014`

The directory structure should look like

```
$DATA/
|–– mscoco2014/
|   |–– train2014/ # contains 1,000 folders like n01440764, n01443537, etc.
|   |–– val2014/
|   |-- captions_train.json
|   |-- coco_karpathy_test.json
|   |-- coco_karpathy_val.json
```

### Flickr

- Create a folder name `flickr/` under `$DATA`.
- Download the dataset form the [Kaggle](https://www.kaggle.com/datasets/eeshawn/flickr30k/data)
- download json file from `https://www.kaggle.com/datasets/wangjilong/dataset-json/` data/flickr/\*.json to `$DATA/flickr`

```
$DATA/
|–– flickr/
|   |–– flickr30k-images/
|   |   |-- *.jpg
|   |-- flickr30k_train.json
|   |-- flickr30k_val.json
|   |-- flickr30k_test.json
```

### FlickrCN

- Create a folder name `flickrcn/` under `$DATA`.
- Download the dataset form the [Kaggle](https://www.kaggle.com/datasets/eeshawn/flickr30k/data)
- download json file from `https://huggingface.co/datasets/OFA-Sys/chinese-clip-eval/resolve/main/Flickr30k-CN.zip` Flickr30k-CN.zip\Flickr30k-CN/\*.jsonl to `$DATA/flickrcn`

```
$DATA/
|–– flickrcn/
|   |–– flickr30k-images/
|   |   |-- *.jpg
|   |-- train_texts.jsonl
|   |-- val_texts.jsonl
|   |-- test_texts.jsonl
```
- Run `generate_flickr30k_cn_json.py --base_dir $DATA/flickrcn/` to generate usable json file
```
$DATA/
|–– flickrcn/
|   |–– flickr30k-images/
|   |   |-- *.jpg
|   |-- flickr30k_cn_train.json
|   |-- flickr30k_cn_val.json
|   |-- flickr30k_cn_test.json
```

### Flickr5k

- Create a folder name `flickr5k/` under `$DATA`.
- Download the dataset form the [Kaggle](https://www.kaggle.com/datasets/wangjilong/self-data/code)
- download json file from `https://www.kaggle.com/datasets/wangjilong/dataset-json/` data/flickr5k/\*.json to `$DATA/flickr5k`

```
$DATA/
|–– flickr5k/
|   |–– flickr5k-images/
|   |   |-- *.jpg
|   |-- flickr5k_train.json.json
|   |-- flickr5k_val.json.json
|   |-- flickr5k_test.json.json
```

## Dataset for AdaCLIP

### MSRVTT

The videos are shared by [Frozen in Time](https://github.com/m-bain/frozen-in-time#-finetuning-benchmarks-msr-vtt):

```
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
```

### DiDeMo

The videos can be downloaded from [LisaAnne/LocalizingMoments](https://github.com/LisaAnne/LocalizingMoments).

### ActivityNet

Download the videos from the [official website](http://activity-net.org/download.html). The authors have made the videos available on Google and Baidu drives.

## Preprocessing

### Frame Extraction

Run `src/adaclip_finetune/utils/frame_extraction.py` after having downloaded the dataset videos and annotations from the website. Make sure that all the videos are in the same directory (no sub-directories allowed).

```
python src/adaclip_finetune/utils/frame_extraction.pyy /path/to/videos /path/to/frames --parallel
```

Subsequently, update the `frames_dir` parameter in the config files `configs/[dataset].json`.

### Annotation Preprocessing

If the videos downloaded differ from the set used in the paper, run `annot_preprocess/{dataset}_preprocess.py` to generate train/test splits used by the dataloader. Splits used in the paper can be found in `annots/`.

To obtain the annotation files used to generate the splits, please download them from the following links:

- MSRVTT annotations are from [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip):

```
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
```

- ActivityNet annotations are from the [project page](https://cs.stanford.edu/people/ranjaykrishna/densevid/) of ActivityNet Captions:

```
wget https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip
```

- DiDeMo annotations have two components: annotations from the [original author](https://github.com/LisaAnne/LocalizingMoments/tree/master/data) and the split used by [Collaborative Experts](https://github.com/albanie/collaborative-experts/tree/master/misc/datasets/didemo).

## Dataset for Qwen2-VL & Qwen2.5-VL Finetune

### ActivityNet-QA

- Please follow https://github.com/MILVLG/activitynet-qa/tree/master to download and seperata train/val dataset

- Then use below python generate_llama_json_limit_frames.py file to generate our train and test dataset:
    ```bash
    python generate_llama_json_limit_frames.py -name val_q -type val -n 500 -seconds 20
    ```

    generate_llama_json_limit_frames.py:

    ```python
    import json
    import os
    import argparse
    import ffmpeg

    # Define the path to the directory where the video files are stored
    video_directory = "where to find dataset"


    def get_video_duration(video_path):
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
            return float(video_stream["duration"])
        except Exception as e:
            print(f"Error getting duration for video {video_path}: {e}")
            return 0


    if __name__ == "__main__":
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Generate LLaMA JSON")
        parser.add_argument("-name", type=str, default="train_q_3000", help="Number of questions to process")
        parser.add_argument("-type", type=str, default="train", help="data type")
        parser.add_argument("-fps", type=float, default=0.2, help="data type")
        parser.add_argument("-n", type=int, default=250, help="data type")
        parser.add_argument("-seconds", type=int, default=20, help="minimum video duration in seconds")
        args = parser.parse_args()
        fps = args.fps
        basic_seconds = args.seconds
        question_json = "../activitynet-qa/dataset/{}.json".format(args.name)
        answer_json = "../activitynet-qa/dataset/{}_a.json".format(args.type)
        combine_json = "../data/activitynet_qa_{}_{}_limit_{}s.json".format(args.type, args.n, basic_seconds)
        print("combine_json:", combine_json)

        # Supported video file extensions
        video_extensions = (".mp4", ".mkv", "webm")

        # Load the questions and answers JSON files
        with open(question_json, "r") as question_file:
            questions = json.load(question_file)

        with open(answer_json, "r") as answer_file:
            answers = json.load(answer_file)

        # Create a dictionary to map question_id to answer for quick lookup
        answer_lookup = {answer["question_id"]: answer for answer in answers}

        combined_data = []
        len_pairs = len(questions)
        # Process each question and look for a corresponding answer
        for question in questions:
            question_id = question["question_id"]
            if question_id in answer_lookup:
                answer = answer_lookup[question_id]

                # Extract the video name typically between 'v_' and the second underscore or end
                video_name_without_path = ("_").join(question_id.split("_")[:-1])
                # Search for the video file that matches the extracted name
                video_path = None
                find_flag = False
                # Walk through the directory to find matching video files
                for root, dirs, files in os.walk(video_directory):
                    for file in files:
                        if file.startswith(video_name_without_path) and file.endswith(video_extensions):
                            video_path = os.path.join(root, file)
                            find_flag = True
                            break
                    if video_path:
                        break
                if not find_flag:
                    print("!!not find:", video_name_without_path)
                if video_path:
                    video_duration = get_video_duration(video_path)
                    if video_duration > basic_seconds:
                        combined_entry = {
                            "messages": [
                                {"content": f"<video>{question['question']}?", "role": "user"},
                                {"content": answer["answer"], "role": "assistant"},
                            ],
                            "videos": [video_path],
                        }
                        combined_data.append(combined_entry)
                        if len(combined_data) % 100 == 0:
                            print(f"Processed {len(combined_data)} entries")
                        if len(combined_data) >= args.n:
                            break
                    else:
                        print("video_duration < basic_seconds", video_duration, video_path)
        # Write the combined data to the output JSON file
        with open(combine_json, "w") as combine_file:
            json.dump(combined_data, combine_file, indent=4)
    ```
### MLVU
- We use the Video Summary part of the MLVU dataset to finetune Qwen2.5-VL 32B.
- Please follow https://huggingface.co/datasets/MLVU/MVLU to download dataset videos and json files.
- Use data in json/9_summary.json to finetune Qwen2.5-VL 32B for video summary.

- Then use below `MLVU_transform.py` file to convert 9_summary.json to Xtune accepted json format, you can change the train and test/val ratio as you need :
    ```bash
    python MLVU_transform.py --train_ratio 0.8 --test_ratio 0.1
    ```
    MLVU_transform.py:
    ```python
    import json
    import random
    import argparse


    def transform_data(input_path, output_path, video_path):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        output = []
        for item in data:
            # Skip invalid items
            if not all(k in item for k in ("question", "answer", "video")):

                continue
            entry = {
                "messages": [
                    {"role": "user", "content": f"<video>{item['question']}"},
                    {"role": "assistant", "content": item["answer"]}
                ],
                "videos": [f"{video_path}{item['video']}"]
            }
            output.append(entry)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

        print(f"Conversion completed, output file: {output_path}")
        return output

    def split_train_test(data, train_ratio, train_path, test_path):
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be in the range (0, 1)")
        n_train = int(len(data) * train_ratio)
        idx = list(range(len(data)))
        random.shuffle(idx)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        train_data = [data[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=4)
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(train_data)} training samples to {train_path} (ratio: {train_ratio:.1%})")
        print(f"Saved {len(test_data)} test samples to {test_path} (ratio: {1-train_ratio:.1%})")

    def split_train_eval_test(data, train_ratio, eval_ratio, train_path, eval_path, test_path):
        if train_ratio + eval_ratio >= 1:
            raise ValueError(f"train_ratio ({train_ratio}) + eval_ratio ({eval_ratio}) = {train_ratio + eval_ratio} must be less than 1")
        if not (0 < train_ratio < 1 and 0 < eval_ratio < 1):
            raise ValueError("train_ratio and eval_ratio must be in the range (0, 1)")
        n_train = int(len(data) * train_ratio)
        n_eval = int(len(data) * eval_ratio)
        idx = list(range(len(data)))
        random.shuffle(idx)
        train_idx = idx[:n_train]
        eval_idx = idx[n_train:n_train + n_eval]
        test_idx = idx[n_train + n_eval:]
        train_data = [data[i] for i in train_idx]
        eval_data = [data[i] for i in eval_idx]
        test_data = [data[i] for i in test_idx]
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=4)
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=4)
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=4)
        test_ratio = 1 - train_ratio - eval_ratio
        print(f"Saved {len(train_data)} training samples to {train_path} (ratio: {train_ratio:.1%})")
        print(f"Saved {len(eval_data)} evaluation samples to {eval_path} (ratio: {eval_ratio:.1%})")
        print(f"Saved {len(test_data)} test samples to {test_path} (ratio: {test_ratio:.1%})")


    if __name__ == "__main__":

        parser = argparse.ArgumentParser(description="Convert and split MLUS data into train/eval/test sets.")
        parser.add_argument("--input_path", type=str, default="./MLVU/json/9_summary.json", help="Input MLUS json path")
        parser.add_argument("--convert_path", type=str, default="MLUS_Vsum_converted.json", help="Output converted json path")
        parser.add_argument("--video_path", type=str, default="./MLVU/video", help="Video directory prefix (with trailing slash)")
        parser.add_argument("--train_ratio", type=float, required=True, help="Ratio of samples for training set (0-1)")
        parser.add_argument("--eval_ratio", type=float, default=None, help="Ratio of samples for evaluation set (0-1). If None, do two-way split")
        parser.add_argument("--train_path", type=str, default="./MLUS_Vsum_train.json", help="Output path for train set")
        parser.add_argument("--eval_path", type=str, default="./MLUS_Vsum_eval.json", help="Output path for eval set")
        parser.add_argument("--test_path", type=str, default="./MLUS_Vsum_test.json", help="Output path for test set")
        args = parser.parse_args()

        output = transform_data(args.input_path, args.convert_path, args.video_path)
        print(f"\nTotal data samples: {len(output)}\n")

        if args.eval_ratio is not None:
            # Three-way split
            split_train_eval_test(output, args.train_ratio, args.eval_ratio, args.train_path, args.eval_path, args.test_path)
        else:
            # Two-way split
            split_train_test(output, args.train_ratio, args.train_path, args.test_path)

    ```
## Update dataset_info.json and dataset json files

**Configuration Requirements:**
- **CLIP/CN-CLIP/AdaCLIP:** Add the corresponding JSON filename only
- **Qwen-VL series:** Requires additional detailed configuration data as below.
### dataset_info.json

```json
{
  "caltech101": {
    "file_name": "caltech101.json"
  },
  "ActivityNet": {
    "file_name": "ActivityNet.json"
  },
  "activitynet_qa_2000_limit_20s": {
    "file_name": "activitynet_qa_2000_limit_20s.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "videos": "videos"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  },
    "MLUS_Vsum_train": {
    "file_name": "MLUS_Vsum_train.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "videos": "videos"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

### caltech101.json

```json
[]
```

### ActivityNet.json

```json
[]
```

### activitynet_qa_2000_limit_20s.json

Generate by `generate_llama_json_limit_frames.py`

### MLUS_Vsum_train.json

Generate by `MLVU_transform.py`