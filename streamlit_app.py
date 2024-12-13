#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
#파일 이름 streamlit_app.py
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1aJolsqI-BlGW0DJhnN3HmWVQTIsB07A3'

# Google Drive에서 파일 다운로드 함수
#@st.cache(allow_output_mutation=True)
@st.cache_resource
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_container_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_container_width=True)
    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://ibb.co/vhvG7bt",
            "https://ibb.co/mbRhtHG",
            "https://ibb.co/f950jrL"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=vgxo53sn8K0",
            "https://www.youtube.com/watch?v=UghqrDn6VqA",
            "https://www.youtube.com/watch?v=uD-M1el6tqQ"
        ],
        'texts': [
            "<실소> 소개편",
            "<실소> 예고편",
            "<실소> ost"
        ]
    },
    labels[1]: {
        'images': [
            "https://ibb.co/xYgPJc5",
            "https://ibb.co/80Kv7d5",
            "https://ibb.co/S51fhWw"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=SmGOlfAEmrI",
            "https://www.youtube.com/watch?v=3U267b-yyFQ",
            "https://www.youtube.com/watch?v=qf2n0QY2N_0"
        ],
        'texts': [
            "<아적반파남우> 몰아보기",
            "<아적반파남우> 1화",
            "<아적반파남우> ost 심월&진철원"
        ]
    },
    labels[2]: {
        'images': [
            "https://ibb.co/rvfzZ0H",
            "https://ibb.co/9cbmTtZ",
            "https://ibb.co/GptRhBh"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=ktA9IUTOWag",
            "https://www.youtube.com/watch?v=adHur3fhGwA",
            "https://www.youtube.com/watch?v=sonY2pmZ-zQ"
        ],
        'texts': [
            "<아친애적소결벽> 하이라이트",
            "<아친애적소결벽> 1화",
            "<아친애적소결벽> ost"
        ]
    },
    labels[3]: {
        'images': [
            "https://ibb.co/2NChpHt",
            "https://ibb.co/dBdXqWv",
            "https://ibb.co/wSrmycZ"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=lFX49tAjmj4",
            "https://www.youtube.com/watch?v=RHsWM1_gStg",
            "https://www.youtube.com/watch?v=3fZbirq9FPg"
        ],
        'texts': [
            "<유성화원2018> 둥산차이♡다오밍스",
            "<유성화원2018> 넷플릭스 예고편",
            "<유성화원2018> ost"
        ]
    },
    labels[4]: {
        'images': [
            "https://ibb.co/qgZDRqS",
            "https://ibb.co/SV3RpSC",
            "https://ibb.co/Y2GD3Zw"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=a29X0rQ2hmA",
            "https://www.youtube.com/watch?v=ReRLQ9GN2iQ",
            "https://www.youtube.com/watch?v=7I1SPKwTXJ0"
        ],
        'texts': [
            "<치아문단순적소미호> 소개편",
            "<치아문단순적소미호> 1화",
            "<치아문단순적소미호> ost ."
        ]
    }
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

