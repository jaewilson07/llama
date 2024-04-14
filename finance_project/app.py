from openai import OpenAI
import streamlit as st
import instructor

import finance_project.finance as fn

client = OpenAI(
    api_key="not needed",
    base_url="http://localhost:8000/v1",
)

client = instructor.patch(client=client)


st.title("🚀 Fake OpenAI Server App")

prompt = st.chat_input("Enter your prompt here")

if prompt:
    st.chat_message("user").markdown(prompt)

    res_price = client.chat.completions.create(
        model="mistral-function-calling",
        messages=[{"role": "user", "content": prompt}],
        # stream=True,
        response_model=fn.ResponseModel,
    )

    try:
        prices = fn.get_stock_prices(ticker=res_price.ticker, days=res_price.days)

        st.json(prices)

        price_prompt = f"{prompt} \n  prices: {prices}"

        res_full = client.chat.completions.create(
            model="mixtral",
            messages=[{"role": "user", "content": price_prompt}],
            stream=True,
        )

        with st.chat_message("ai") as chat:
            completed_message = ""
            message = st.empty()

            for chunk in res_full:
                content = chunk.choices[0].delta.content

                if not content:
                    continue

                completed_message += content
                message.markdown(completed_message)

    except Exception as e:
        st.chat_message("ai").markdown(e)
