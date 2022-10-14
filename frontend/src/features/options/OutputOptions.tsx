import { Flex } from '@chakra-ui/react';

import HiresOptions from './HiresOptions';
import SeamlessOptions from './SeamlessOptions';

const OutputOptions = () => {

  return (
    <Flex gap={2} direction={'column'}>
      <SeamlessOptions />
      <HiresOptions />
    </Flex>
  );
};

export default OutputOptions;
