import { Flex } from '@chakra-ui/layout';
import React, { ChangeEvent } from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAISwitch from '../../../../common/components/IAISwitch';
import { setShouldUseInitImage } from '../../optionsSlice';

export default function ImageToImageAccordion() {
  const dispatch = useAppDispatch();

  const initialImagePath = useAppSelector(
    (state: RootState) => state.options.initialImagePath
  );

  const shouldUseInitImage = useAppSelector(
    (state: RootState) => state.options.shouldUseInitImage
  );

  const handleChangeShouldUseInitImage = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldUseInitImage(e.target.checked));
  return (
    <Flex
      justifyContent={'space-between'}
      alignItems={'center'}
      width={'100%'}
      mr={2}
    >
      <p>Image to Image</p>
      <IAISwitch
        isDisabled={!initialImagePath}
        isChecked={shouldUseInitImage}
        onChange={handleChangeShouldUseInitImage}
      />
    </Flex>
  );
}
