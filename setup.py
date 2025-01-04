from setuptools import setup, find_packages

setup(
    name="ra-aid-webui",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.32.0",
        "watchdog>=3.0.0",
        "python-socketio>=5.11.1",
        "aiohttp>=3.9.3",
        "rich>=13.7.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-timeout>=2.2.0",
            "pytest-mock>=3.10.0",
            "pytest-cov>=4.1.0"
        ]
    }
) 