from ultralytics import YOLO

def v8predict(img,weights):
    # Load a pretrained YOLOv8n model
    model = YOLO(weights)
    # Run inference on the source
    res = model.predict(img, conf=0.1)  # list of Results objects
    resdict = res[0]
    speed = resdict.speed
    pre = speed.get('preprocess',0.0)  
    inference = speed.get('inference', 0.0)
    post = speed.get('postprocess', 0.0)
    timetaken = (pre + inference + post)/1000.0
    res_plotted = res[0].plot()[:, :, ::-1]
    return res_plotted, timetaken


# # Load a pretrained YOLOv8n model
# model = YOLO(r'v8weights\best.pt')
# # Run inference on the source
# res = model.predict('Japan_000082.jpg', conf=0.1) 
# resdict = res[0] # list of Results objects
# speed = resdict.speed
# pre = speed.get('preprocess',0.0)  
# inference = speed.get('inference', 0.0)
# post = speed.get('postprocess', 0.0)
# timetaken = pre + inference + post

# breakpoint()

