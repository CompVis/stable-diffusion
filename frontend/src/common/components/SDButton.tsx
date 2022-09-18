import { Button, ButtonProps } from '@chakra-ui/react';

interface Props extends ButtonProps {
  label: string;
}

/**
 * Reusable customized button component. Originally was more customized - now probably unecessary.
 *
 * TODO: Get rid of this.
 */
const SDButton = (props: Props) => {
  const { label, size = 'sm', ...rest } = props;
  return (
    <Button size={size} {...rest}>
      {label}
    </Button>
  );
};

export default SDButton;
