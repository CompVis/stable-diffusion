import { Flex } from '@chakra-ui/react';
import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/store';
import { setSeamless } from './optionsSlice';
import { ChangeEvent } from 'react';
import IAISwitch from '../../common/components/IAISwitch';

const SeamlessOptions = () => {
  const dispatch = useAppDispatch();

  const seamless = useAppSelector((state: RootState) => state.options.seamless);

  const handleChangeSeamless = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setSeamless(e.target.checked));

  return (
    <Flex gap={2} direction={'column'}>
      <IAISwitch
        label="Seamless tiling"
        fontSize={'md'}
        isChecked={seamless}
        onChange={handleChangeSeamless}
      />
    </Flex>
  );
};

export default SeamlessOptions;