import { Flex } from '@chakra-ui/layout';
import React, { ChangeEvent } from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAISwitch from '../../../../common/components/IAISwitch';
import { setShouldRunESRGAN } from '../../optionsSlice';

export default function Upscale() {
  const isESRGANAvailable = useAppSelector(
    (state: RootState) => state.system.isESRGANAvailable
  );

  const shouldRunESRGAN = useAppSelector(
    (state: RootState) => state.options.shouldRunESRGAN
  );

  const dispatch = useAppDispatch();
  const handleChangeShouldRunESRGAN = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldRunESRGAN(e.target.checked));
  return (
    <Flex
      justifyContent={'space-between'}
      alignItems={'center'}
      width={'100%'}
      mr={2}
    >
      <p>Upscale</p>
      <IAISwitch
        isDisabled={!isESRGANAvailable}
        isChecked={shouldRunESRGAN}
        onChange={handleChangeShouldRunESRGAN}
      />
    </Flex>
  );
}
