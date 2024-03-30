import base64
import io
import this
import random
from datetime import time

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import keras
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.src.layers import Activation
from keras.src.utils import img_to_array, load_img
from keras.utils import to_categorical
import matplotlib.pyplot as plt

if 'Body' not in st.session_state:
    st.session_state['Body'] = []
if 'Type' not in st.session_state:
    st.session_state['Type'] = []
if 'Weather' not in st.session_state:
    st.session_state['Weather'] = []
if 'image_bytes' not in st.session_state:
    st.session_state['image_bytes'] = []
if 'image_name' not in st.session_state:
    st.session_state['image_name'] = []
if "model" not in st.session_state:
    st.session_state['model'] = []



def create_clothes_table():
    column_headers = ['Body', 'Type', 'Weather']


    data = [
        [st.session_state['Body'][i], st.session_state['Type'][i], st.session_state['Weather'][i]]
        for i in range(len(st.session_state['Body']))
    ]
    clothes = pd.DataFrame(data=data,columns=column_headers)
    clothes.index += 1

    return clothes

def display_table(df):
    row_colors = [
        {'selector': 'tr:nth-of-type(odd)',
         'props': [('background', 'linear-gradient(to right, white, lightblue)')]},
        {'selector': 'tr:nth-of-type(even)',
         'props': [('background', 'linear-gradient(to right, lightblue, white)')]}
    ]
    st.table(df.style.set_table_styles([
                                                {'selector': 'th, .index',  # Apply style to both headers and index
                                                 'props': [('background-color', 'deepskyblue'),
                                                           # Red background color
                                                           ('color', 'white'),  # White text color
                                                           ('font-family', 'Arial'),
                                                           ('text-align', 'center')]},  # Center-align text
                                                {'selector': 'td',
                                                 'props': [('font-family', 'Arial'),
                                                           ('text-align', 'center')]}
                                                # Center-align text in data cells
                                            ] + row_colors))




def create_model():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

    return model


def add_to_table(predicted_class, predicted_class_name):
    if predicted_class in [0, 2, 4, 6]:
        Body = 'Upper Body'
    elif predicted_class == 1:
        Body = 'Lower Body'
    elif predicted_class == 3:
        Body = 'Dress'
    else:
        Body = 'Shoes'

    if predicted_class in [0, 3, 5]:
        Weather = 'Hot'
    elif predicted_class in [2, 4, 9]:
        Weather = 'Cold'
    else:
        Weather = 'Both'

    # Creating a dictionary containing the data for the new row
    new_row =  [Body,predicted_class_name,Weather]

    return new_row


# Function to add clothes using Streamlit


def add_clothes():
    global clothes
    st.markdown(
        "<h1 style='text-align: center; color: deepskyblue; font-size: 1.5em; font-family: cursive;'>Upload an image</h1>",
        unsafe_allow_html=True
    )
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Sneaker",
                   "Ankle boot"]

    uploaded_file = st.file_uploader("", type=["jpg", "png"])



    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_bytes = uploaded_file.read()
        img = load_img(uploaded_file, target_size=(28, 28), color_mode="grayscale")
        img_array = img_to_array(img)
        img_array_reshaped = img_array.reshape((28, 28))
        img_array = img_array.astype('float32') / 255
        st.session_state['image_name'].append(file_name)
        st.session_state['image_bytes'].append(file_bytes)


        with st.spinner("In progress..."):
            if not st.session_state['model']:
                st.session_state['model'] = create_model()

        # כאשר התהליך הארוך מסתיים, התצוגה של הספינר תיעלם ויצגו את התוצאה
        st.success('Excellent! The image was identified and uploaded to the clothing database')
        predictions = st.session_state['model'].predict(img_array.reshape((1, 28, 28, 1)))
        predicted_class = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class]
        new_row = add_to_table( predicted_class, predicted_class_name)
        return new_row

def create_page_style():
    hide_scrollbar = """
            <style>
            body {
                overflow: hidden;
            }
            </style>
        """
    st.markdown(hide_scrollbar, unsafe_allow_html=True)
    # Load the background image using st.image()
    # Create a container to overlay other elements on top of the image
    background = '''
            <style>
            .stApp {
                background-color: #f2fafd;;
                background-size: 100% 50%;
            }
            </style>
        '''

    # Apply the background image to the container
    st.markdown(background, unsafe_allow_html=True)

    # Now you can add other Streamlit elements as usual
    st.markdown(
        "<h1 style='text-align: center; color: deepskyblue; font-size: 4.5em; font-family: cursive; text-shadow: 2px 2px 2px #ccc; letter-spacing: 2px;'>What would you like to do?</h1>",
        unsafe_allow_html=True
    )
    for i in range(10):
        st.write("")

    st.markdown(
        '<img src="https://img.freepik.com/premium-photo/blue-room-with-round-table-blue-chair-with-round-table-it_662214-54474.jpg?w=1380" '
        'style="width: 500px; height: 300px; position:  absolute; top: 500%; left: 50%; transform: translate(-50%, -50%);">',
        unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: deepskyblue; /* צבע רקע */
            color: white; /* צבע טקסט */
            padding: 15px 25px;
            text-align: center;
            font-size: 4em;
            display: block; /* שימוש ב-display: block כדי לקבוע את הכפתורים כלוח עם אורך מלא */
            width: fit-content; /* הגדרת הרוחב כהתאם לתוכן */
            margin-top: 150px; /* מרווח למעלה */
            margin-left: auto; /* מרכוז הכפתורים באופן אופקי */
            margin-right: auto; /* מרכוז הכפתורים באופן אופקי */
            border-radius: 80px;
        }
        .stButton>button:hover {
            background-color: steelblue; /* צבע רקע בעת העכבר מעל */
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def remove_row(df):
    df['index'] = df.index - 1
    st.markdown(
        "<h1 style='text-align: center; color: deepskyblue; font-size: 1.5em; font-family: cursive;'>Select category to filter by:</h1>",
        unsafe_allow_html=True
    )

    selected_category = st.multiselect('', df['Type'].unique())
    filtered_indices = []  # Initialize an empty list to store filtered indices

    if selected_category:  # Check if any categories are selected
        for i, row in df.iterrows():  # Iterate through DataFrame rows efficiently
            if row['Type'] in selected_category:
                filtered_indices.append(row['index'])  # Use adjusted index

    for i, image_bytes in enumerate(st.session_state['image_bytes']):
        if i in filtered_indices:  # Filter based on filtered_indices
            image = Image.open(io.BytesIO(image_bytes))
            with st.container():
                image = image.resize((100, 100))
                st.image(image)
                text_to_display = f"item {i + 1}"  # Pre-evaluate the text

                html_code = f"""
                <p style="color:black; font-size:20px; padding-left:20px;">{text_to_display}</p>
                """

                st.write(html_code, unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align: center; color: deepskyblue; font-size: 1.5em; font-family: cursive;'>Select the item you want to delete</h1>",
        unsafe_allow_html=True
    )
    index  = st.radio("",
                         options=[index + 1 for index in filtered_indices],
                         key="image_deletion")  # Use a single radio button for selection
    if st.button('Remove selected item'):
        del st.session_state['Body'][index-1]
        del st.session_state['Type'][index-1]
        del st.session_state['Weather'][index-1]
        del st.session_state['image_bytes'][index-1]
        del st.session_state['image_name'][index-1]
        st.write('<p style="color:deepskyblue; font-size:20px; text-align:center;">Item successfully removed!</p>',
                 unsafe_allow_html=True)
def recommendation(df):
    st.markdown(
        "<h1 style='text-align: center; color: deepskyblue; font-size: 1.5em; font-family: cursive;'>Some small details..</h1>",
        unsafe_allow_html=True
    )


    weather = st.radio("What's the weather ?", ("Hot", "Cold"))
    event = st.radio("What event are you going to ?", ("Party", "Work", "Conference", "University" , "Restaurant", "coffee shop", "Event"))
    Time = st.radio("At what time ?", ("Day", "Night"))
    place = st.radio("Will you be outside or inside ?", ("Outside", "Inside"))

    if st.button('Get a recommendation'):
        if weather == 'Hot':
            df= df[df['Weather'] != 'Cold']
        elif weather == 'Cold':
            df = df[df['Weather'] != 'Hot']
        chosen_option = random.randint(0, 1)

        if chosen_option==0:
            dress_df= df[df['Type']== 'Dress']
            if not dress_df.empty:
                st.markdown(
                    "<h1 style='text-align: center; color: deepskyblue; font-size: 1.5em; font-family: cursive;'>The recommendation is</h1>",
                    unsafe_allow_html=True
                )
                dress_row = dress_df.sample()
                image = Image.open(io.BytesIO( st.session_state['image_bytes'][dress_row.index[0]-1]))
                with st.container():
                    image = image.resize((100, 100))
                    st.image(image)
                shoes_df = df[df['Type'] == 'Sneaker']
                if not shoes_df.empty:
                    shoes_row = shoes_df.sample()
                    image = Image.open(io.BytesIO(st.session_state['image_bytes'][shoes_row.index[0] - 1]))
                    with st.container():
                        image = image.resize((100, 100))
                        st.image(image)
                else:
                    st.write("Missing shoes in the database")



            else:
                chosen_option = 1


        if chosen_option == 1:
            st.markdown(
                "<h1 style='text-align: center; color: deepskyblue; font-size: 1.5em; font-family: cursive;'>The recommendation is</h1>",
                unsafe_allow_html=True
            )
            upper_df = df[df['Body']=='Upper Body']
            if not upper_df.empty:
                upper_row = upper_df.sample()
                image = Image.open(io.BytesIO(st.session_state['image_bytes'][upper_row.index[0] - 1]))
                with st.container():
                    image = image.resize((100, 100))
                    st.image(image)
            else:
                st.write("Missing top part in the database")
            lower_df= df[df['Body']== 'Lower Body']
            if not lower_df.empty:
                lower_row= lower_df.sample()
                image = Image.open(io.BytesIO(st.session_state['image_bytes'][lower_row.index[0]-1]))
                with st.container():
                    image = image.resize((100, 100))
                    st.image(image)
            else:
                st.write("Missing lower part in the database")
            shoes_df = df[df['Body'] == 'Shoes']
            if not shoes_df.empty:
                shoes_row = shoes_df.sample()
                image = Image.open(io.BytesIO(st.session_state['image_bytes'][shoes_row.index[0]-1]))
                with st.container():
                    image = image.resize((100, 100))
                    st.image(image)
            else:
                st.write("Missing shoes in the database")


def main_page():
    create_page_style()
    col1, col2, col3 ,col4= st.columns(4)
    with col1:
        if st.button("Added a new item"):
            st.session_state['make_set'] = False
            st.session_state['remove_item'] =  False
            st.session_state['data'] = False
            st.session_state['add_item'] = True

    with col2:
        if st.button("Remove a item"):
            st.session_state['add_item'] = False
            st.session_state['make_set'] = False
            st.session_state['data'] = False
            st.session_state['remove_item'] = True


    with col3:
        if st.button("Make a set!"):
            st.session_state['remove_item'] = False
            st.session_state['add_item'] = False
            st.session_state['data'] = False
            st.session_state['make_set'] = True

    with col4:
        if st.button("clothing database"):
            st.session_state['remove_item'] = False
            st.session_state['add_item'] = False
            st.session_state['make_set'] = False
            st.session_state['data'] = True

    if 'add_item' in st.session_state and st.session_state['add_item']:
        new_row= add_clothes()
        if new_row is not None:
            st.session_state['Body'].append(new_row[0])
            st.session_state['Type'].append(new_row[1])
            st.session_state['Weather'].append(new_row[2])
            clothes=create_clothes_table()
            display_table(clothes)


    if 'remove_item' in st.session_state and st.session_state['remove_item']:
        clothes = create_clothes_table()
        remove_row(clothes)

    if 'make_set' in st.session_state and st.session_state['make_set']:
        clothes = create_clothes_table()
        recommendation(clothes)

    if 'data' in st.session_state and st.session_state['data']:
        clothes = create_clothes_table()
        display_table(clothes)




# קריאה לפונקציה הראשית
if __name__ == '__main__':
    main_page()
