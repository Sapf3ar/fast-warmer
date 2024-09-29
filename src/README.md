semantic_candidates.py
Description
Generates video neighbors based on semantic similarity of video titles and descriptions using sentence embeddings.

Usage
```
python semantic_candidates.py --parquet_path PATH_TO_VIDEO_STAT_PARQUET --model_name MODEL_NAME --max_seq_len MAX_SEQ_LEN --device DEVICE --k K
```
Arguments
--parquet_path: Path to the video statistics Parquet file.
--model_name: Name of the SentenceTransformer model (default: 'intfloat/multilingual-e5-base').
--max_seq_len: Maximum sequence length for the model (default: 256).
--device: Device to run the model on ('cuda' or 'cpu', default: 'cuda').
--k: Number of nearest neighbors to search for (default: 300).

```
python semantic_candidates.py --parquet_path data/video_stat.parquet --k 300
stat_candidates.py
```

Description
Finds video neighbors based on statistical features of videos.

Usage
```
python stat_candidates.py --parquet_path PATH_TO_VIDEO_STAT_PARQUET --k K --index_path INDEX_PATH
```
Arguments
--parquet_path: Path to the video statistics Parquet file.
--k: Number of nearest neighbors to search for (default: 300).
--index_path: Path to save or load the FAISS index (default: 'stat_video_index.faiss').

```
python stat_candidates.py --parquet_path data/video_stat.parquet --k 300
als_candidates.py
```
Description
Trains an ALS model and recommends videos based on similar authors.

Usage
```
python als_candidates.py --video_stats_path VIDEO_STATS_PATH --logs_path_pattern LOGS_PATH_PATTERN --model_path MODEL_PATH [--train] [--evaluate] [--author_id AUTHOR_ID]
```
Arguments
--video_stats_path: Path to the video statistics Parquet file.
--logs_path_pattern: Glob pattern to the logs Parquet files.
--model_path: Path to save or load the ALS model (default: 'ALS_model.npz').
--train: Include this flag to train the model.
--evaluate: Include this flag to evaluate the model.
--author_id: Author ID for which to recommend videos.
Example
To train the model:

```
python als_candidates.py --video_stats_path data/video_stat.parquet --logs_path_pattern 'data/logs*.parquet' --train
```

To evaluate the model:

```
python als_candidates.py --video_stats_path data/video_stat.parquet --logs_path_pattern 'data/logs*.parquet' --evaluate
````

To get recommendations for an author:

````
python als_candidates.py --video_stats_path data/video_stat.parquet --logs_path_pattern 'data/logs*.parquet' --author_id 12345
````