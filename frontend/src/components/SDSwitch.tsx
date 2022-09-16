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

const SDSwitch = (props: Props) => {
  const {
    label,
    isDisabled = false,
    fontSize = 'md',
    size = 'md',
    width,
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
            whiteSpace='nowrap'
          >
            {label}
          </FormLabel>
        )}
        <Switch size={size} {...rest} />
      </Flex>
    </FormControl>
  );
};

export default SDSwitch;
