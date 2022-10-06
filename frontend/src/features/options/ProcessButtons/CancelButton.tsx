import React from 'react';
import { MdCancel } from 'react-icons/md';
import { cancelProcessing } from '../../../app/socketio/actions';
import { useAppDispatch, useAppSelector } from '../../../app/store';
import IAIIconButton from '../../../common/components/IAIIconButton';
import { systemSelector } from '../../../common/hooks/useCheckParameters';
import { useHotkeys } from 'react-hotkeys-hook';

export default function CancelButton() {
  const dispatch = useAppDispatch();
  const { isProcessing, isConnected } = useAppSelector(systemSelector);
  const handleClickCancel = () => dispatch(cancelProcessing());

  useHotkeys(
    'shift+x',
    () => {
      if (isConnected || isProcessing) {
        handleClickCancel();
      }
    },
    [isConnected, isProcessing]
  );

  return (
    <IAIIconButton
      icon={<MdCancel />}
      tooltip="Cancel"
      aria-label="Cancel"
      isDisabled={!isConnected || !isProcessing}
      onClick={handleClickCancel}
      className="cancel-btn"
    />
  );
}
