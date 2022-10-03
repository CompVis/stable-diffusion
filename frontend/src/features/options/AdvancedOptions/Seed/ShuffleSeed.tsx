import { Button } from '@chakra-ui/react';
import React from 'react';
import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from '../../../../app/constants';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import randomInt from '../../../../common/util/randomInt';
import { setSeed } from '../../optionsSlice';

export default function ShuffleSeed() {
  const dispatch = useAppDispatch();
  const shouldRandomizeSeed = useAppSelector(
    (state: RootState) => state.options.shouldRandomizeSeed
  );

  const handleClickRandomizeSeed = () =>
    dispatch(setSeed(randomInt(NUMPY_RAND_MIN, NUMPY_RAND_MAX)));

  return (
    <Button
      size={'sm'}
      isDisabled={shouldRandomizeSeed}
      onClick={handleClickRandomizeSeed}
    >
      <p>Shuffle</p>
    </Button>
  );
}
