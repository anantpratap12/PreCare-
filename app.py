from src import about, braintumor, home, info, mail, pneumonia
import streamlit as st
import os

def init():
    st.session_state.page = 'Homepage'
    st.session_state.project = False
    st.session_state.model = False

    st.session_state.pages = {
        'Homepage': home.main,
        'About Us': about.main,
        'Contact Us': mail.main,
        'Pneumonia Detection': pneumonia.main,
        'Brain Tumor Detection': braintumor.main,
        'About the Dataset': info.main
    }

def draw_style():

    style = """
        <style>
        .stApp {background-image: url("");
background-size: cover;
background-repeat: no-repeat;
background-position: center;

}
            header {visibility: visible;}
            footer {visibility: hidden;} 
            
            
        </style>
    """

    st.set_page_config(page_title='PreCare Disease Detection Model',)
    
    st.markdown(style, unsafe_allow_html=True)

def load_page():
    st.session_state.pages[st.session_state.page]()

def set_page(loc=None, reset=False):
    if not st.session_state.page == 'Homepage':
        for key in list(st.session_state.keys()):
            if key not in ('page', 'project', 'model', 'pages', 'set'):
                st.session_state.pop(key)

    if loc:
        st.session_state.page = loc
    else:
        st.session_state.page = st.session_state.set

    if reset:
        st.session_state.project = False
    elif st.session_state.page in ('Message me', 'About me'):
        st.session_state.project = True
        st.session_state.model = False
    else:
        pass

def change_button():
    set_page('Pneumonia Detection')
    st.session_state.model = True
    st.session_state.project = True

def prev():
    st.header("Disease Detection Deep Learning Models")

    models = ["Tumor Detection","Heart Detection","Any other Detection"]
    models_info = ["Info about tumor Detection","Info about heart detection","Info about other detection"]
    press = [False]*len(models)
    with st.sidebar:
        st.title("Browse Models")
        for i,model in enumerate(models):
            press[i] = st.sidebar.button(model)
            with st.expander("See Info"):
                st.write(models_info[i])

def main():
    if 'page' not in st.session_state or 'pages' not in st.session_state:
        init()

    draw_style()

    with st.sidebar:
        project, about ,contact= st.columns([0.8, 1, 1.2])

        if not st.session_state.project:
            project.button('Models', on_click=change_button)
        else:
            project.button('Home', on_click=set_page, args=('Homepage', True))

        if st.session_state.project and st.session_state.model:
            st.radio(
                'Models',
                ['Pneumonia Detection','Brain Tumor Detection'],
                key='set',
                on_change=set_page,
            )

        about.button('About Us', on_click=set_page, args=('About Us',))

        contact.button(
            'Contact Us', on_click=set_page, args=('Contact Us',)
        )
        st.button("About the Dataset",on_click=set_page,args=("About the Dataset",))
        
    if st.session_state.page in ['Homepage', 'About the Dataset']:
        with st.sidebar:
            img_path = "test_files/p1.jpeg"
            img2_path = "test_files/bt1.jpeg"
            if os.path.exists(img_path):
                st.image(img_path)
            else:
                st.warning("No image available.")
            
            if os.path.exists(img2_path):
                st.image(img2_path)
            else:
                st.warning("No image available.")
        
    load_page()

if __name__ == '__main__':
    main()
