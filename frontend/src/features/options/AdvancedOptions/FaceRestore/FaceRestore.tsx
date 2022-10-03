import { Flex } from '@chakra-ui/layout';
import React, { ChangeEvent } from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAISwitch from '../../../../common/components/IAISwitch';
import { setShouldRunGFPGAN } from '../../optionsSlice';

export default function FaceRestore() {
  const isGFPGANAvailable = useAppSelector(
    (state: RootState) => state.system.isGFPGANAvailable
  );

  const shouldRunGFPGAN = useAppSelector(
    (state: RootState) => state.options.shouldRunGFPGAN
  );

  const dispatch = useAppDispatch();

  const handleChangeShouldRunGFPGAN = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldRunGFPGAN(e.target.checked));

  return (
    <Flex
      justifyContent={'space-between'}
      alignItems={'center'}
      width={'100%'}
      mr={2}
    >
      <p>Restore Face</p>
      <IAISwitch
        isDisabled={!isGFPGANAvailable}
        isChecked={shouldRunGFPGAN}
        onChange={handleChangeShouldRunGFPGAN}
      />
    </Flex>
  );
}
