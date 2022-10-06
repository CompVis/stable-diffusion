import { Button, useToast } from '@chakra-ui/react';
import React, { useCallback } from 'react';
import { FileRejection } from 'react-dropzone';
import { useAppDispatch } from '../../app/store';
import ImageUploader from '../../features/options/ImageUploader';

interface InvokeImageUploaderProps {
  label?: string;
  icon?: any;
  onMouseOver?: any;
  OnMouseout?: any;
  dispatcher: any;
  styleClass?: string;
}

export default function InvokeImageUploader(props: InvokeImageUploaderProps) {
  const { label, icon, dispatcher, styleClass, onMouseOver, OnMouseout } =
    props;

  const toast = useToast();
  const dispatch = useAppDispatch();

  // Callbacks to for handling file upload attempts
  const fileAcceptedCallback = useCallback(
    (file: File) => dispatch(dispatcher(file)),
    [dispatch, dispatcher]
  );

  const fileRejectionCallback = useCallback(
    (rejection: FileRejection) => {
      const msg = rejection.errors.reduce(
        (acc: string, cur: { message: string }) => acc + '\n' + cur.message,
        ''
      );

      toast({
        title: 'Upload failed',
        description: msg,
        status: 'error',
        isClosable: true,
      });
    },
    [toast]
  );

  return (
    <ImageUploader
      fileAcceptedCallback={fileAcceptedCallback}
      fileRejectionCallback={fileRejectionCallback}
      styleClass={styleClass}
    >
      <Button
        size={'sm'}
        fontSize={'md'}
        fontWeight={'normal'}
        onMouseOver={onMouseOver}
        onMouseOut={OnMouseout}
        leftIcon={icon}
        width={'100%'}
      >
        {label ? label : null}
      </Button>
    </ImageUploader>
  );
}
