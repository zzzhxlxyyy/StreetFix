import streamlit as st 
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title ="StreetFix"
)
st.title("StreetFix")
st.markdown("##### *Paving the Way to Smarter Roads – Detecting Damage, Ensuring Safety*")
st_lottie(
        "https://lottie.host/48c4848d-6192-4121-ae71-bed484e6c7b0/1neHJTEJIu.json"
    )

st.divider()

st.markdown("### Background")
st.write(""" 
With over 1.4 billion automobiles globally, road stress poses challenges to safety, infrastructure, and the economy. 
Increased vehicular usage elevates accident risks and road damage, straining resources. Prioritizing transportation 
infrastructure upgrades is vital for sustained growth. Employing artificial intelligence and computer vision, essential 
object detection algorithms must be implemented for precise road condition assessment and targeted maintenance. Addressing 
these concerns, the StreetFix project emerges as a solution for efficient road damage detection, enhancing both safety and 
economic resilience.
""")

st.sidebar.success("Select a page above")

st.divider()

st.markdown("### Problem Statement")
lottie_cols = st.columns([0.05, 0.25,0.05, 0.25, 0.05, 0.25, 0.05])
with lottie_cols[1]:
    st_lottie(
        "https://lottie.host/de6f5df1-b9bc-4756-8d1e-84f7697b00a6/gqorkTuScc.json"
    )
with lottie_cols[3]:
    st_lottie(
        "https://lottie.host/a4fb1d0a-71ce-4023-812a-8edbc8cd65a7/wb2z1Uuhum.json"
    )
with lottie_cols[5]:
    st_lottie(
        "https://lottie.host/d6c93847-cdcf-4291-bf20-baeeffd451db/EoR8OVGaXu.json"
    )
obj_cols = st.columns(
    [0.05, 0.25,0.05, 0.25, 0.05, 0.25, 0.05],
)
with obj_cols[1]:
    st.markdown(
        "<p class='center-p'> Time-consuming and labor-intensive </p>", unsafe_allow_html=True
    )
with obj_cols[3]:
    st.markdown(
        "<p class='center-p'> Cover very small regions ; ccuracy varies amongst inspectors </p>", unsafe_allow_html=True
    )
with obj_cols[5]:
    st.markdown(
        "<p class='center-p'> Not performed on a regular basis, causing delayed discovery</p>", unsafe_allow_html=True
    )

st.divider()

st.markdown("### Objectives")
lottie_cols = st.columns([0.05, 0.25,0.05, 0.25, 0.05, 0.25, 0.05])
with lottie_cols[1]:
    st_lottie(
        "https://lottie.host/2a57abd4-448a-468e-8341-19da4e63b460/SZGSguHIQD.json"
    )
with lottie_cols[3]:
    st_lottie(
        "https://lottie.host/bcd1c3df-c55d-4b61-8f61-acd96663eab7/IMeQdF0Fbn.json"
    )
with lottie_cols[5]:
    st_lottie(
        "https://lottie.host/94b03e5f-997d-4903-823f-2d512bf988f4/DgpsfA4Ccd.json"
    )
obj_cols = st.columns(
    [0.05, 0.25,0.05, 0.25, 0.05, 0.25, 0.05],
)
with obj_cols[1]:
    st.markdown(
        "<p class='center-p'> Automation of the road damage detection </p>", unsafe_allow_html=True
    )
with obj_cols[3]:
    st.markdown(
        "<p class='center-p'> Evaluate the performance of model </p>", unsafe_allow_html=True
    )
with obj_cols[5]:
    st.markdown(
        "<p class='center-p'> Present a dashboard </p>", unsafe_allow_html=True
    )

# st.write("""
#     Problem Statement
#     Road damage detection has been done manually by humans in most countries.Human inspectors physically evaluate road surfaces for problems such as potholes, and cracks in traditional techniques. However, manual inspections may be time-consuming and labor-intensive. Inspectors cover very small regions, and detection is based on the inspector's knowledge and expertise. Visual examinations are also subjective, and damage detection accuracy varies amongst inspectors.Moreover, manual inspections may not be performed on a regular basis, resulting in the delayed discovery of new road damage or the worsening of existing problems. As manual inspections are labor-intensive, they can be costly in terms of both persons and time. Other than manual detection, some urban administrators use complex sensors to detect road damage. Both methods stated are costly and require a huge amount of human effort to detect the damage.
         
#     Objectives
#     1. To develop automation of the road damage detection from images.
#     2. To evaluate the performance of the road damage detection model.
#     3. To present a dashboard for road damage detection.

# """)

# normal_text("""
            
#     Problem Statement
#     1. Manual inspections may be time-consuming and labor-intensive
#     2. Inspectors cover very small regions. Damage detection accuracy varies amongst inspectors
#     3. Manual inspections may not be performed on a regular basis, resulting in the delayed discovery of road damage.
         
#     Objectives
#     1. To develop automation of the road damage detection from images.
#     2. To evaluate the performance of the road damage detection model.
#     3. To present a dashboard for road damage detection.

# """)

         

