if 没有关闭 wandb:
    创建一个 wandb 实验
    项目名 = args.wandb_project
    实验名 = args.wandb_run_name
    保存所有命令行参数
    并且把几个路径参数转成字符串后保存


wandb.log({"loss": loss.item()})

wandb 就会把 loss 曲线记录到这个 run 下面。