import streamlit as st
from emotions import detect_attention
from personal_advice import generate_advice

advice_argument = []


def main():
    global advice_argument

    st.title("学生注意力分析系统")
    mode = st.sidebar.selectbox(
        label="请您选择模式",
        options=("注意力监测", "个性化建议")
    )
    if mode == "注意力监测":
        if st.button("开始分析专注情况"):
            distracted_ratio = detect_attention()
            advice_argument.append(distracted_ratio)
    elif mode == "个性化建议":
        if st.button("咨询专家获得个性化建议"):
            generate_advice(advice_argument)


if __name__ == "__main__":
    main()
