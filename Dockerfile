FROM continuumio/miniconda3

WORKDIR /src/sarsen

COPY environment.yml /src/sarsen/

RUN conda install -c conda-forge gcc python=3.11 \
    && conda env update -n base -f environment.yml

COPY . /src/sarsen

RUN pip install --no-deps -e .
