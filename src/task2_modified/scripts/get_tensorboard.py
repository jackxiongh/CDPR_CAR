from tensorboard.backend.event_processing import event_accumulator
import os, sys

#加载日志数据
tb_path = os.path.abspath(os.path.join(sys.path[0], "../../../../tensorboard/SAC_4/events.out.tfevents.1659866638.handleandwheel-ubuntu.1124524.0"))
ea=event_accumulator.EventAccumulator(tb_path) 
ea.Reload()
# print(ea.scalars.Keys())

val_psnr=ea.scalars.Items('rollout/ep_rew_mean')
# print(len(val_psnr))
for i in val_psnr:
    print("%d\t%f" % (i.step, i.value))