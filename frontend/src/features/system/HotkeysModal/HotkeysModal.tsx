import {
  Modal,
  ModalCloseButton,
  ModalContent,
  ModalOverlay,
  useDisclosure,
} from '@chakra-ui/react';
import React, { cloneElement, ReactElement } from 'react';
import HotkeysModalItem from './HotkeysModalItem';

type HotkeysModalProps = {
  /* The button to open the Settings Modal */
  children: ReactElement;
};

export default function HotkeysModal({ children }: HotkeysModalProps) {
  const {
    isOpen: isHotkeyModalOpen,
    onOpen: onHotkeysModalOpen,
    onClose: onHotkeysModalClose,
  } = useDisclosure();

  const hotkeys = [
    { title: 'Invoke', desc: 'Generate an image', hotkey: 'Ctrl+Enter' },
    { title: 'Cancel', desc: 'Cancel image generation', hotkey: 'Shift+X' },
    {
      title: 'Toggle Gallery',
      desc: 'Open and close the gallery drawer',
      hotkey: 'G',
    },
    {
      title: 'Set Seed',
      desc: 'Use the seed of the current image',
      hotkey: 'S',
    },
    {
      title: 'Set Parameters',
      desc: 'Use all parameters of the current image',
      hotkey: 'A',
    },
    { title: 'Restore Faces', desc: 'Restore the current image', hotkey: 'R' },
    { title: 'Upscale', desc: 'Upscale the current image', hotkey: 'U' },
    {
      title: 'Show Info',
      desc: 'Show metadata info of the current image',
      hotkey: 'I',
    },
    {
      title: 'Send To Image To Image',
      desc: 'Send the current image to Image to Image module',
      hotkey: 'Shift+I',
    },
    { title: 'Delete Image', desc: 'Delete the current image', hotkey: 'Del' },
    {
      title: 'Focus Prompt',
      desc: 'Focus the prompt input area',
      hotkey: 'Alt+A',
    },
    {
      title: 'Previous Image',
      desc: 'Display the previous image in the gallery',
      hotkey: 'Arrow left',
    },
    {
      title: 'Next Image',
      desc: 'Display the next image in the gallery',
      hotkey: 'Arrow right',
    },
    {
      title: 'Change Tabs',
      desc: 'Switch to another workspace',
      hotkey: '1-6',
    },
    {
      title: 'Theme Toggle',
      desc: 'Switch between dark and light modes',
      hotkey: 'Shift+D',
    },
    {
      title: 'Console Toggle',
      desc: 'Open and close console',
      hotkey: '`',
    },
  ];

  const renderHotkeyModalItems = () => {
    const hotkeyModalItemsToRender: ReactElement[] = [];

    hotkeys.forEach((hotkey, i) => {
      hotkeyModalItemsToRender.push(
        <HotkeysModalItem
          key={i}
          title={hotkey.title}
          description={hotkey.desc}
          hotkey={hotkey.hotkey}
        />
      );
    });

    return hotkeyModalItemsToRender;
  };

  return (
    <>
      {cloneElement(children, {
        onClick: onHotkeysModalOpen,
      })}
      <Modal isOpen={isHotkeyModalOpen} onClose={onHotkeysModalClose}>
        <ModalOverlay />
        <ModalContent className="hotkeys-modal">
          <ModalCloseButton />
          <h1>Keyboard Shorcuts</h1>
          <div className="hotkeys-modal-items">{renderHotkeyModalItems()}</div>
        </ModalContent>
      </Modal>
    </>
  );
}
