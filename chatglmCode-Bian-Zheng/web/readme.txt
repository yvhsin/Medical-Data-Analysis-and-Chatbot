下载模型
>> git clone https://huggingface.co/THUDM/chatglm-6b

启动anaconda环境

安装依赖
>> pip install -r requirements.txt

解压finetune.zip中的output

修改start_web.py中的output路径

运行服务（记得start_web.py和chatglm-6b保持同一目录）
>> python start_web.py