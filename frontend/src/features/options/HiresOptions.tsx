import { Flex } from '@chakra-ui/react';
import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/store';
import { setHiresFix } from './optionsSlice';
import { ChangeEvent } from 'react';
import IAISwitch from '../../common/components/IAISwitch';

/**
 * Image output options. Includes width, height, seamless tiling.
 */
const HiresOptions = () => {
  const dispatch = useAppDispatch();

  const hiresFix = useAppSelector((state: RootState) => state.options.hiresFix);

  const handleChangeHiresFix = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setHiresFix(e.target.checked));


  return (
    <Flex gap={2} direction={'column'}>
      <IAISwitch
        label="High Res Optimization"
        fontSize={'md'}
        isChecked={hiresFix}
        onChange={handleChangeHiresFix}
      />
    </Flex>
  );
};

export default HiresOptions;
