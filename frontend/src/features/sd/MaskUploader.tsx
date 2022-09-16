import { useToast } from '@chakra-ui/react';
import { cloneElement, ReactElement, SyntheticEvent, useCallback } from 'react';
import { FileRejection, useDropzone } from 'react-dropzone';
import { useAppDispatch } from '../../app/hooks';
import { uploadMaskImage } from '../../app/socketio';

type Props = {
    children: ReactElement;
};

const MaskUploader = ({ children }: Props) => {
    const dispatch = useAppDispatch();
    const toast = useToast();

    const onDrop = useCallback(
        (acceptedFiles: Array<File>, fileRejections: Array<FileRejection>) => {
            fileRejections.forEach((rejection: FileRejection) => {
                const msg = rejection.errors.reduce(
                    (acc: string, cur: { message: string }) =>
                        acc + '\n' + cur.message,
                    ''
                );

                toast({
                    title: 'Upload failed',
                    description: msg,
                    status: 'error',
                    isClosable: true,
                });
            });

            acceptedFiles.forEach((file: File) => {
                dispatch(uploadMaskImage(file));
            });
        },
        [dispatch, toast]
    );

    const { getRootProps, getInputProps, open } = useDropzone({
        onDrop,
        accept: {
            'image/jpeg': ['.jpg', '.jpeg', '.png'],
        },
    });

    const handleClickUploadIcon = (e: SyntheticEvent) => {
        e.stopPropagation();
        open();
    };

    return (
        <div {...getRootProps()}>
            <input {...getInputProps({ multiple: false })} />
            {cloneElement(children, {
                onClick: handleClickUploadIcon,
            })}
        </div>
    );
};

export default MaskUploader;
