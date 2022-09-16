import { createAction, Middleware } from '@reduxjs/toolkit';
import { io } from 'socket.io-client';
import {
    addImage,
    clearIntermediateImage,
    removeImage,
    SDImage,
    SDMetadata,
    setGalleryImages,
    setIntermediateImage,
} from '../features/gallery/gallerySlice';
import {
    addLogEntry,
    setCurrentStep,
    setIsConnected,
    setIsProcessing,
} from '../features/system/systemSlice';
import { v4 as uuidv4 } from 'uuid';
import { setInitialImagePath, setMaskPath } from '../features/sd/sdSlice';
import {
    backendToFrontendParameters,
    frontendToBackendParameters,
} from './parameterTranslation';

export interface SocketIOResponse {
    status: 'OK' | 'ERROR';
    message?: string;
    data?: any;
}

export const socketioMiddleware = () => {
    const { hostname, port } = new URL(window.location.href);

    const socketio = io(`http://${hostname}:9090`);

    let areListenersSet = false;

    const middleware: Middleware = (store) => (next) => (action) => {
        const { dispatch, getState } = store;
        if (!areListenersSet) {
            // CONNECT
            socketio.on('connect', () => {
                try {
                    dispatch(setIsConnected(true));
                } catch (e) {
                    console.error(e);
                }
            });

            // DISCONNECT
            socketio.on('disconnect', () => {
                try {
                    dispatch(setIsConnected(false));
                    dispatch(setIsProcessing(false));
                    dispatch(addLogEntry(`Disconnected from server`));
                } catch (e) {
                    console.error(e);
                }
            });

            // PROCESSING RESULT
            socketio.on(
                'result',
                (data: {
                    url: string;
                    type: 'generation' | 'esrgan' | 'gfpgan';
                    uuid?: string;
                    metadata: { [key: string]: any };
                }) => {
                    try {
                        const newUuid = uuidv4();
                        const { type, url, uuid, metadata } = data;
                        switch (type) {
                            case 'generation': {
                                const translatedMetadata =
                                    backendToFrontendParameters(metadata);
                                dispatch(
                                    addImage({
                                        uuid: newUuid,
                                        url,
                                        metadata: translatedMetadata,
                                    })
                                );
                                dispatch(
                                    addLogEntry(`Image generated: ${url}`)
                                );

                                break;
                            }
                            case 'esrgan': {
                                const originalImage =
                                    getState().gallery.images.find(
                                        (i: SDImage) => i.uuid === uuid
                                    );
                                const newMetadata = {
                                    ...originalImage.metadata,
                                };
                                newMetadata.shouldRunESRGAN = true;
                                newMetadata.upscalingLevel =
                                    metadata.upscale[0];
                                newMetadata.upscalingStrength =
                                    metadata.upscale[1];
                                dispatch(
                                    addImage({
                                        uuid: newUuid,
                                        url,
                                        metadata: newMetadata,
                                    })
                                );
                                dispatch(
                                    addLogEntry(`ESRGAN upscaled: ${url}`)
                                );

                                break;
                            }
                            case 'gfpgan': {
                                const originalImage =
                                    getState().gallery.images.find(
                                        (i: SDImage) => i.uuid === uuid
                                    );
                                const newMetadata = {
                                    ...originalImage.metadata,
                                };
                                newMetadata.shouldRunGFPGAN = true;
                                newMetadata.gfpganStrength =
                                    metadata.gfpgan_strength;
                                dispatch(
                                    addImage({
                                        uuid: newUuid,
                                        url,
                                        metadata: newMetadata,
                                    })
                                );
                                dispatch(
                                    addLogEntry(`GFPGAN fixed faces: ${url}`)
                                );

                                break;
                            }
                        }
                        dispatch(setIsProcessing(false));
                    } catch (e) {
                        console.error(e);
                    }
                }
            );

            // PROGRESS UPDATE
            socketio.on('progress', (data: { step: number }) => {
                try {
                    dispatch(setIsProcessing(true));
                    dispatch(setCurrentStep(data.step));
                } catch (e) {
                    console.error(e);
                }
            });

            // INTERMEDIATE IMAGE
            socketio.on(
                'intermediateResult',
                (data: { url: string; metadata: SDMetadata }) => {
                    try {
                        const uuid = uuidv4();
                        const { url, metadata } = data;
                        dispatch(
                            setIntermediateImage({
                                uuid,
                                url,
                                metadata,
                            })
                        );
                        dispatch(
                            addLogEntry(`Intermediate image generated: ${url}`)
                        );
                    } catch (e) {
                        console.error(e);
                    }
                }
            );

            // ERROR FROM BACKEND
            socketio.on('error', (message) => {
                try {
                    dispatch(addLogEntry(`Server error: ${message}`));
                    dispatch(setIsProcessing(false));
                    dispatch(clearIntermediateImage());
                } catch (e) {
                    console.error(e);
                }
            });

            areListenersSet = true;
        }

        // HANDLE ACTIONS

        switch (action.type) {
            // GENERATE IMAGE
            case 'socketio/generateImage': {
                dispatch(setIsProcessing(true));
                dispatch(setCurrentStep(-1));

                const {
                    generationParameters,
                    esrganParameters,
                    gfpganParameters,
                } = frontendToBackendParameters(
                    getState().sd,
                    getState().system
                );

                socketio.emit(
                    'generateImage',
                    generationParameters,
                    esrganParameters,
                    gfpganParameters
                );

                dispatch(
                    addLogEntry(
                        `Image generation requested: ${JSON.stringify({
                            ...generationParameters,
                            ...esrganParameters,
                            ...gfpganParameters,
                        })}`
                    )
                );
                break;
            }

            // RUN ESRGAN (UPSCALING)
            case 'socketio/runESRGAN': {
                const imageToProcess = action.payload;
                dispatch(setIsProcessing(true));
                dispatch(setCurrentStep(-1));
                const { upscalingLevel, upscalingStrength } = getState().sd;
                const esrganParameters = {
                    upscale: [upscalingLevel, upscalingStrength],
                };
                socketio.emit('runESRGAN', imageToProcess, esrganParameters);
                dispatch(
                    addLogEntry(
                        `ESRGAN upscale requested: ${JSON.stringify({
                            file: imageToProcess.url,
                            ...esrganParameters,
                        })}`
                    )
                );
                break;
            }

            // RUN GFPGAN (FIX FACES)
            case 'socketio/runGFPGAN': {
                const imageToProcess = action.payload;
                dispatch(setIsProcessing(true));
                dispatch(setCurrentStep(-1));
                const { gfpganStrength } = getState().sd;

                const gfpganParameters = {
                    gfpgan_strength: gfpganStrength,
                };
                socketio.emit('runGFPGAN', imageToProcess, gfpganParameters);
                dispatch(
                    addLogEntry(
                        `GFPGAN fix faces requested: ${JSON.stringify({
                            file: imageToProcess.url,
                            ...gfpganParameters,
                        })}`
                    )
                );
                break;
            }

            // DELETE IMAGE
            case 'socketio/deleteImage': {
                const imageToDelete = action.payload;
                const { url } = imageToDelete;
                socketio.emit(
                    'deleteImage',
                    url,
                    (response: SocketIOResponse) => {
                        if (response.status === 'OK') {
                            dispatch(removeImage(imageToDelete));
                            dispatch(addLogEntry(`Image deleted: ${url}`));
                        }
                    }
                );
                break;
            }

            // GET ALL IMAGES FOR GALLERY
            case 'socketio/requestAllImages': {
                socketio.emit(
                    'requestAllImages',
                    (response: SocketIOResponse) => {
                        dispatch(setGalleryImages(response.data));
                        dispatch(
                            addLogEntry(`Loaded ${response.data.length} images`)
                        );
                    }
                );
                break;
            }

            // CANCEL PROCESSING
            case 'socketio/cancelProcessing': {
                socketio.emit('cancel', (response: SocketIOResponse) => {
                    const { intermediateImage } = getState().gallery;
                    if (response.status === 'OK') {
                        dispatch(setIsProcessing(false));
                        if (intermediateImage) {
                            dispatch(addImage(intermediateImage));
                            dispatch(
                                addLogEntry(
                                    `Intermediate image saved: ${intermediateImage.url}`
                                )
                            );

                            dispatch(clearIntermediateImage());
                        }
                        dispatch(addLogEntry(`Processing canceled`));
                    }
                });
                break;
            }

            // UPLOAD INITIAL IMAGE
            case 'socketio/uploadInitialImage': {
                const file = action.payload;

                socketio.emit(
                    'uploadInitialImage',
                    file,
                    file.name,
                    (response: SocketIOResponse) => {
                        if (response.status === 'OK') {
                            dispatch(setInitialImagePath(response.data));
                            dispatch(
                                addLogEntry(
                                    `Initial image uploaded: ${response.data}`
                                )
                            );
                        }
                    }
                );
                break;
            }

            // UPLOAD MASK IMAGE
            case 'socketio/uploadMaskImage': {
                const file = action.payload;

                socketio.emit(
                    'uploadMaskImage',
                    file,
                    file.name,
                    (response: SocketIOResponse) => {
                        if (response.status === 'OK') {
                            dispatch(setMaskPath(response.data));
                            dispatch(
                                addLogEntry(
                                    `Mask image uploaded: ${response.data}`
                                )
                            );
                        }
                    }
                );
                break;
            }
        }

        next(action);
    };

    return middleware;
};

// Actions to be used by app

export const generateImage = createAction<undefined>('socketio/generateImage');
export const runESRGAN = createAction<SDImage>('socketio/runESRGAN');
export const runGFPGAN = createAction<SDImage>('socketio/runGFPGAN');
export const deleteImage = createAction<SDImage>('socketio/deleteImage');
export const requestAllImages = createAction<undefined>(
    'socketio/requestAllImages'
);
export const cancelProcessing = createAction<undefined>(
    'socketio/cancelProcessing'
);
export const uploadInitialImage = createAction<File>(
    'socketio/uploadInitialImage'
);
export const uploadMaskImage = createAction<File>('socketio/uploadMaskImage');
