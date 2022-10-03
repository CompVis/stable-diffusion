import { Flex } from '@chakra-ui/react';
import SeedWeights from './SeedWeights';
import VariationAmount from './VariationAmount';

/**
 * Seed & variation options. Includes iteration, seed, seed randomization, variation options.
 */
const VariationsOptions = () => {
  return (
    <Flex gap={2} direction={'column'}>
      <VariationAmount />
      <SeedWeights />
    </Flex>
  );
};

export default VariationsOptions;
