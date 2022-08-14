FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

COPY environment.yml /install/environment.yml
RUN conda env update -f /install/environment.yml

# Create a non-root user and switch to it
WORKDIR /app
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

RUN chmod -R 777 /app

COPY --chown=user:user . /app

VOLUME /results

CMD "/bin/bash"