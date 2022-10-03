import {
  Flex,
  FormControl,
  FormLabel,
  Switch,
  SwitchProps,
} from '@chakra-ui/react';

interface Props extends SwitchProps {
  label?: string;
  width?: string | number;
}

/**
 * Customized Chakra FormControl + Switch multi-part component.
 */
const IAISwitch = (props: Props) => {
  const {
    label,
    isDisabled = false,
    fontSize = 'md',
    size = 'md',
    width = 'auto',
    ...rest
  } = props;
  return (
    <FormControl isDisabled={isDisabled} width={width}>
      <Flex justifyContent={'space-between'} alignItems={'center'}>
        {label && (
          <FormLabel
            fontSize={fontSize}
            marginBottom={1}
            flexGrow={2}
            whiteSpace="nowrap"
          >
            {label}
          </FormLabel>
        )}
        <Switch size={size} className="switch-button" {...rest} />
      </Flex>
    </FormControl>
  );
};

export default IAISwitch;
