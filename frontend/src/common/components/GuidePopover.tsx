import {
  Popover,
  PopoverArrow,
  PopoverContent,
  PopoverTrigger,
  Box,
} from '@chakra-ui/react';
import { SystemState } from '../../features/system/systemSlice';
import { useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import { createSelector } from '@reduxjs/toolkit';
import { ReactElement } from 'react';
import { Feature, FEATURES } from '../../app/features';

type GuideProps = {
  children: ReactElement;
  feature: Feature;
};

const systemSelector = createSelector(
  (state: RootState) => state.system,
  (system: SystemState) => system.shouldDisplayGuides
);

const GuidePopover = ({ children, feature }: GuideProps) => {
  const shouldDisplayGuides = useAppSelector(systemSelector);
  const { text } = FEATURES[feature];
  return shouldDisplayGuides ? (
    <Popover trigger={'hover'}>
      <PopoverTrigger>
        <Box>{children}</Box>
      </PopoverTrigger>
      <PopoverContent
        className={`guide-popover-content`}
        maxWidth="400px"
        onClick={(e) => e.preventDefault()}
        cursor={'initial'}
      >
        <PopoverArrow className="guide-popover-arrow" />
        <div className="guide-popover-guide-content">{text}</div>
      </PopoverContent>
    </Popover>
  ) : (
    <></>
  );
};

export default GuidePopover;
