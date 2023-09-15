import streamlit as st

st.sidebar.title('Cosmetic Recommend')
st.sidebar.header('추천받고 싶은 유형을 선택하세요')
if st.sidebar.checkbox("상품명 입력"):
    product = st.sidebar.text_input(label="상품명", value="default value")
    if st.sidebar.button("추천받기"):
        st.header(f"{product}와 유사한 제품입니다.")
        selected_product = st.selectbox('궁금한 제품을 선택하세요.', ['1.제품','2.제품','3.제품','4.제품','5.제품'])
        st.write('선택하신 제품은 {}입니다.'.format(selected_product))


if st.sidebar.checkbox("고객타입 입력"):
    gender = st.sidebar.selectbox("성별",["남성","여성"])
    age = st.sidebar.selectbox("나이",["1020대","30대","40대","50대 이상"])
    skintype = st.sidebar.selectbox("피부타입",["복합성","건성","수분부족지성","지성","중성","극건성"])
    skintrouble = st.sidebar.selectbox("피부트러블",["민감성","건조함","탄력없음","트러블","주름","모공","칙칙함","복합성"])
    if st.sidebar.button("추천받기"):
        st.header(f"{gender}, {age}, {skintype}, {skintrouble} 타입 고객님께 추천하는 제품입니다.")



