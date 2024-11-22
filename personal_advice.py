from http import HTTPStatus
import dashscope
from dashscope import Generation
import random
import streamlit as st


def generate_advice(argument=[0]):
    if argument == []:
        argument = [0]
    distracted_ratio = argument[0]

    st.write("您的个性化建议如下：")

    dashscope.api_key = ""
    question = f"我是一位语文老师，在听学生复述课文，考察他对课文的理解。然而，学生在复述课文时出现了注意力不集中的情况，注意力不集中的时间占总复述时间的比例是{distracted_ratio:.2%}，比例越小说明学生越专注。请你以教育专家的口吻，直接与这位同学进行对话。请你针对他的注意力不集中占比等情况，提出一些个性化的、特别的建议来提高这位学生的注意力集中程度。"
    messages = [{'role': 'system', 'content': 'You are an education expert with over 10 years experience.'},
                {'role': 'user', 'content': question}]
    response = Generation.call(model="qwen-long",
                               messages=messages,
                               # 设置随机数种子seed，如果没有设置，则随机数种子默认为1234
                               seed=random.randint(1, 10000),
                               # 将输出设置为"message"格式
                               result_format='message')
    assistant_output = response.output.choices[0]['message']['content']
    if response.status_code == HTTPStatus.OK:
        st.write(assistant_output)
        print(assistant_output)
    else:
        st.error('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


if __name__ == '__main__':
    generate_advice()
