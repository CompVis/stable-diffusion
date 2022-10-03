import { IconButtonProps, IconButton, Tooltip } from '@chakra-ui/react';

interface Props extends IconButtonProps {
  tooltip?: string;
}

/**
 * Reusable customized button component. Originally was more customized - now probably unecessary.
 *
 * TODO: Get rid of this.
 */
const IAIIconButton = (props: Props) => {
  const { tooltip = '', onClick, ...rest } = props;
  return (
    <Tooltip label={tooltip}>
      <IconButton {...rest} cursor={onClick ? 'pointer' : 'unset'} onClick={onClick}/>
    </Tooltip>
  );
};

export default IAIIconButton;
