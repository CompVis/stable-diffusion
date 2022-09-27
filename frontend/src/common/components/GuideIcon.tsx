import { Box, forwardRef, Icon } from '@chakra-ui/react';
import { IconType } from 'react-icons';
import { MdHelp } from 'react-icons/md';
import { Feature } from '../../app/features';
import GuidePopover from './GuidePopover';

type GuideIconProps = {
  feature: Feature;
  icon?: IconType;
};

const GuideIcon = forwardRef(
  ({ feature, icon = MdHelp }: GuideIconProps, ref) => (
    <GuidePopover feature={feature}>
      <Box ref={ref}>
        <Icon as={icon} />
      </Box>
    </GuidePopover>
  )
);

export default GuideIcon;
