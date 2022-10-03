import { Flex } from '@chakra-ui/react';
import React from 'react';
import GenerateVariations from './GenerateVariations';

export default function Variations() {
  return (
    <Flex
      justifyContent={'space-between'}
      alignItems={'center'}
      width={'100%'}
      mr={2}
    >
      <p>Variations</p>
      <GenerateVariations />
    </Flex>
  );
}
