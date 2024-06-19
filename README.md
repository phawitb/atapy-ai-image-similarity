# atapy-ai-image-similarity

### run
```
uvicorn imgsimilarity_api:app --reload --port 8003
```
#### encode image
```
POST : http://127.0.0.1:8003/atapy-image-similarity/encode
{
    "img": ["https://static.amarintv.com/images/upload/editor/source/BuM2023/389807.jpg","https://static.amarintv.com/images/upload/editor/source/BuM2023/389807.jpg"]
}
```

#### compare 2 image
```
POST : http://127.0.0.1:8003/atapy-image-similarity/compare
{
    "img": ["https://static.amarintv.com/images/upload/editor/source/BuM2023/389807.jpg","https://static.amarintv.com/images/upload/editor/source/BuM2023/389807.jpg"]
}
```
