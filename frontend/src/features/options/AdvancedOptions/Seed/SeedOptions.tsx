import { Flex } from '@chakra-ui/react';
import RandomizeSeed from './RandomizeSeed';
import Seed from './Seed';
import ShuffleSeed from './ShuffleSeed';

/**
 * Seed & variation options. Includes iteration, seed, seed randomization, variation options.
 */
const SeedOptions = () => {
  return (
    <Flex gap={2} direction={'column'}>
      <RandomizeSeed />
      <Flex gap={2}>
        <Seed />
        <ShuffleSeed />
      </Flex>
    </Flex>
  );
};

export default SeedOptions;
