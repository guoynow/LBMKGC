# Large Model-Driven Balanced Multimodal Knowledge Graph Completion



##  Data preparation

>We use the MMKG datasets proposed in [MMRNS](https://github.com/quqxui/MMRNS). 
>
>We use the SDXL and CLIP provided by [Hugging Face](https://huggingface.co/).
>
>You need to download the [embeddings](https://drive.google.com/drive/folders/1WBt7UjDFgkIRawRmgEEpX4GW8U6A9l_P), and place them in the directories "./embeddings".

## Dependencies

> - torch==2.2.2
> - numpy==2.1.2
> - scikit-learn==1.6.1
> - tqdm==4.64.1
> - Our code is based on OpenKE, an open-source KGC project. You can refer to the [OpenKE repo](https://github.com/thunlp/OpenKE) to build the environment.

## Train and Evaluation
>You should run the script `mmkgc/make.sh` to ensure that the `release/Base.so` file is compatible with your environment.
>
>You can use the shell scripts to conduct the experiments. 
>
>```shell
>python Main_LBMKGC.py -dataset MKG-W -margin 16 -epoch 1000 -save MKG-W-checkpoint -learning_rate 1e-5
>
>python Main_LBMKGC.py -dataset MKG-Y -margin 24 -epoch 1250 -save MKG-Y-checkpoint -learning_rate 1e-5
>
>python Main_LBMKGC.py -dataset DB15K -margin 12 -epoch 1250 -save DB15K-checkpoint -learning_rate 2e-5
>```
>

