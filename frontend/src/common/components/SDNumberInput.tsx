import {
  FormControl,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Text,
  FormLabel,
  NumberInputProps,
  Flex,
} from '@chakra-ui/react';

interface Props extends NumberInputProps {
  label?: string;
  width?: string | number;
}

/**
 * Customized Chakra FormControl + NumberInput multi-part component.
 */
const SDNumberInput = (props: Props) => {
  const {
    label,
    isDisabled = false,
    fontSize = 'md',
    size = 'sm',
    width,
    isInvalid,
    ...rest
  } = props;
  return (
    <FormControl isDisabled={isDisabled} width={width} isInvalid={isInvalid}>
      <Flex gap={2} justifyContent={'space-between'} alignItems={'center'}>
        {label && (
          <FormLabel marginBottom={1}>
            <Text fontSize={fontSize} whiteSpace="nowrap">
              {label}
            </Text>
          </FormLabel>
        )}
        <NumberInput
          size={size}
          {...rest}
          keepWithinRange={false}
          clampValueOnBlur={true}
        >
          <NumberInputField fontSize={'md'} />
          <NumberInputStepper>
            <NumberIncrementStepper />
            <NumberDecrementStepper />
          </NumberInputStepper>
        </NumberInput>
      </Flex>
    </FormControl>
  );
};

export default SDNumberInput;
