import {
    Flex,
    FormControl,
    FormLabel,
    Heading,
    HStack,
    Modal,
    ModalBody,
    ModalCloseButton,
    ModalContent,
    ModalFooter,
    ModalHeader,
    ModalOverlay,
    Switch,
    Text,
    useDisclosure,
} from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import {
    setShouldConfirmOnDelete,
    setShouldDisplayInProgress,
    SystemState,
} from './systemSlice';
import { RootState } from '../../app/store';
import SDButton from '../../components/SDButton';
import { persistor } from '../../main';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { cloneElement, ReactElement } from 'react';

const systemSelector = createSelector(
    (state: RootState) => state.system,
    (system: SystemState) => {
        const { shouldDisplayInProgress, shouldConfirmOnDelete } = system;
        return { shouldDisplayInProgress, shouldConfirmOnDelete };
    },
    {
        memoizeOptions: { resultEqualityCheck: isEqual },
    }
);

type Props = {
    children: ReactElement;
};

const SettingsModal = ({ children }: Props) => {
    const {
        isOpen: isSettingsModalOpen,
        onOpen: onSettingsModalOpen,
        onClose: onSettingsModalClose,
    } = useDisclosure();

    const {
        isOpen: isRefreshModalOpen,
        onOpen: onRefreshModalOpen,
        onClose: onRefreshModalClose,
    } = useDisclosure();

    const { shouldDisplayInProgress, shouldConfirmOnDelete } =
        useAppSelector(systemSelector);

    const dispatch = useAppDispatch();

    const handleClickResetWebUI = () => {
        persistor.purge().then(() => {
            onSettingsModalClose();
            onRefreshModalOpen();
        });
    };

    return (
        <>
            {cloneElement(children, {
                onClick: onSettingsModalOpen,
            })}

            <Modal isOpen={isSettingsModalOpen} onClose={onSettingsModalClose}>
                <ModalOverlay />
                <ModalContent>
                    <ModalHeader>Settings</ModalHeader>
                    <ModalCloseButton />
                    <ModalBody>
                        <Flex gap={5} direction='column'>
                            <FormControl>
                                <HStack>
                                    <FormLabel marginBottom={1}>
                                        Display in-progress images (slower)
                                    </FormLabel>
                                    <Switch
                                        isChecked={shouldDisplayInProgress}
                                        onChange={(e) =>
                                            dispatch(
                                                setShouldDisplayInProgress(
                                                    e.target.checked
                                                )
                                            )
                                        }
                                    />
                                </HStack>
                            </FormControl>
                            <FormControl>
                                <HStack>
                                    <FormLabel marginBottom={1}>
                                        Confirm on delete
                                    </FormLabel>
                                    <Switch
                                        isChecked={shouldConfirmOnDelete}
                                        onChange={(e) =>
                                            dispatch(
                                                setShouldConfirmOnDelete(
                                                    e.target.checked
                                                )
                                            )
                                        }
                                    />
                                </HStack>
                            </FormControl>

                            <Heading size={'md'}>Reset Web UI</Heading>
                            <Text>
                                Resetting the web UI only resets the browser's
                                local cache of your images and remembered
                                settings. It does not delete any images from
                                disk.
                            </Text>
                            <Text>
                                If images aren't showing up in the gallery or
                                something else isn't working, please try
                                resetting before submitting an issue on GitHub.
                            </Text>
                            <SDButton
                                label='Reset Web UI'
                                colorScheme='red'
                                onClick={handleClickResetWebUI}
                            />
                        </Flex>
                    </ModalBody>

                    <ModalFooter>
                        <SDButton
                            label='Close'
                            onClick={onSettingsModalClose}
                        />
                    </ModalFooter>
                </ModalContent>
            </Modal>

            <Modal
                closeOnOverlayClick={false}
                isOpen={isRefreshModalOpen}
                onClose={onRefreshModalClose}
                isCentered
            >
                <ModalOverlay bg='blackAlpha.300' backdropFilter='blur(40px)' />
                <ModalContent>
                    <ModalBody pb={6} pt={6}>
                        <Flex justifyContent={'center'}>
                            <Text fontSize={'lg'}>
                                Web UI has been reset. Refresh the page to
                                reload.
                            </Text>
                        </Flex>
                    </ModalBody>
                </ModalContent>
            </Modal>
        </>
    );
};

export default SettingsModal;
