import { Button, ButtonProps, Tooltip } from '@chakra-ui/react';

interface Props extends ButtonProps {
  label: string;
  tooltip?: string;
}

/**
 * Reusable customized button component. Originally was more customized - now probably unecessary.
 *
 * TODO: Get rid of this.
 */
const IAIButton = (props: Props) => {
  const { label, tooltip = '', size = 'sm', ...rest } = props;
  return (
    <Tooltip label={tooltip}>
      <Button size={size} {...rest}>
        {label}
      </Button>
    </Tooltip>
  );
};

export default IAIButton;
