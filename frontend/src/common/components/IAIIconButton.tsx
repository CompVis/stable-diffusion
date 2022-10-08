import {
  IconButtonProps,
  IconButton,
  Tooltip,
  PlacementWithLogical,
} from '@chakra-ui/react';

interface Props extends IconButtonProps {
  tooltip?: string;
  tooltipPlacement?: PlacementWithLogical | undefined;
}

/**
 * Reusable customized button component. Originally was more customized - now probably unecessary.
 *
 * TODO: Get rid of this.
 */
const IAIIconButton = (props: Props) => {
  const { tooltip = '', tooltipPlacement = 'bottom', onClick, ...rest } = props;
  return (
    <Tooltip label={tooltip} hasArrow placement={tooltipPlacement}>
      <IconButton
        {...rest}
        cursor={onClick ? 'pointer' : 'unset'}
        onClick={onClick}
      />
    </Tooltip>
  );
};

export default IAIIconButton;
