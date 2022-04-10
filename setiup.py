from setuptools import setup

setup(name='fair_vae',
      version='0.3.0',
      description='Evidential Deep learning implementation Pytorch',
      url='https://github.com/Tuttusa/FairVAE',
      author='Vaunorage',
      author_email='vaunorage@tuttusa.io',
      license='MIT',
      packages=['fair_vae'],
      install_requires=[
          "pandas==1.4.2",
          "numpy==1.22.3",
          "torch==1.11.0",
          "pytorch-lightning==1.6.0",
          "pydantic==1.9.0",
          "torchmetrics==0.7.3",
          "scikit-learn==1.0.2",
          "seaborn==0.11.2",
          "rdt==0.6.4"
      ],
      zip_safe=False)
