from setuptools import setup, find_packages

setup(
    name='rag-insight',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "llama_index",
        "llmsherpa",
        "streamlit",
        "python-dotenv==1.0.1"
    ],
    author="Ioana Dragan",
    author_email="draganioana.23@gmail.com",
    description="RAG Insight: A Streamlit RAG chatbot powered by OpenAI, LLamaIndex and LLM Sherpa",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license="MIT",
    url="https://github.com/IoanaDragan/rag-insight",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],

)