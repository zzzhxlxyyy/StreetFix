import streamlit as st 
from PIL import Image
import assets.DataPreprocess as DataPreprocess
import yolov7.yolov7_predict as yolov7_predict
from io import BytesIO
import yolov8.yolov8_predict as yolov8_predict
# import frcnn.frcnn_predict as frcnn_predict
import time
import numpy as np
import hydralit_components as hc
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title ="StreetFix"
)

titlecol = st.columns([0.8,0.2])
with titlecol[0]:
    st.title("Where is the road damage?")
with titlecol[1]:
    st_lottie(
        "https://lottie.host/9935262d-9c04-4254-ac80-9f629271f82e/ZS2X37K8NK.json"
    )

img = st.file_uploader("", type=["jpg", "png"])
if img is None:
    st.text("Please upload an image file")
else:
    img = img.read()
    img = Image.open(BytesIO(img))
    img = DataPreprocess.resizeimg(img)
    img_np = np.array(img)
    # img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # st.image(img, use_column_width=True)

    if st.button("Detect"):
        with hc.HyLoader('Detecting...',hc.Loaders.pacman):
            time.sleep(5)
            v7pred, v7time = yolov7_predict.detect(img, weights=r'yolov7\v7weights\best.pt', conf_thres=0.1, nosave=True)
            v8pred, v8time = yolov8_predict.v8predict(img, r'yolov8\v8weights\best.pt')
            # t0 = time.time()
            # frcnnpred = frcnn_predict.frcnnpredict(img_np ,r'frcnn\checkpoint\best_coco_bbox_mAP_epoch_20.pth')
            # frcnntime = time.time() - t0
            # frcnnpred_rgb = cv2.cvtColor(frcnnpred, cv2.COLOR_BGR2RGB)
        st.success('Done!')

        col1, col2, col3 = st.columns(3)
        
        # with col1:
        #     st.image(frcnnpred,
        #             caption='Faster RCNN',
        #             use_column_width=True
        #             )
        #     st.write(f"Time taken : {frcnntime:.3f}s")
        with col2:
            st.image(v7pred[0], 
                    caption='Yolov7',
                    use_column_width=True)
            st.write(f"Time taken : {v7time:.3f}s")
        with col3:
            st.image(v8pred,
                    caption='Yolov8',
                    use_column_width=True
                    )
            st.write(f"Time taken : {v8time:.3f}s")



        


