import {
  IconButtonProps,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Text,
  useDisclosure,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import {
  cloneElement,
  ReactElement,
  SyntheticEvent,
} from 'react';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { deleteImage } from '../../app/socketio';
import { RootState } from '../../app/store';
import SDButton from '../../components/SDButton';
import { setShouldConfirmOnDelete, SystemState } from '../system/systemSlice';
import { SDImage } from './gallerySlice';

interface Props extends IconButtonProps {
  image: SDImage;
  'aria-label': string;
  children: ReactElement;
}

const systemSelector = createSelector(
  (state: RootState) => state.system,
  (system: SystemState) => system.shouldConfirmOnDelete
);

/*
TODO: The modal and button to open it should be two different components,
but their state is closely related and I'm not sure how best to accomplish it.
*/
const DeleteImageModalButton = (props: Omit<Props, 'aria-label'>) => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const dispatch = useAppDispatch();
  const shouldConfirmOnDelete = useAppSelector(systemSelector);

  const handleClickDelete = (e: SyntheticEvent) => {
    e.stopPropagation();
    shouldConfirmOnDelete ? onOpen() : handleDelete();
  };

  const { image, children } = props;

  const handleDelete = () => {
    dispatch(deleteImage(image));
    onClose();
  };

  const handleDeleteAndDontAsk = () => {
    dispatch(deleteImage(image));
    dispatch(setShouldConfirmOnDelete(false));
    onClose();
  };

  return (
    <>
      {cloneElement(children, {
        onClick: handleClickDelete,
      })}

      <Modal isOpen={isOpen} onClose={onClose}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Are you sure you want to delete this image?</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Text>It will be deleted forever!</Text>
          </ModalBody>

          <ModalFooter justifyContent={'space-between'}>
            <SDButton label={'Yes'} colorScheme='red' onClick={handleDelete} />
            <SDButton
              label={"Yes, and don't ask me again"}
              colorScheme='red'
              onClick={handleDeleteAndDontAsk}
            />
            <SDButton label='Cancel' colorScheme='blue' onClick={onClose} />
          </ModalFooter>
        </ModalContent>
      </Modal>
    </>
  );
};

export default DeleteImageModalButton;
