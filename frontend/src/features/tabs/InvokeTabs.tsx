import { Tab, TabPanel, TabPanels, Tabs, Tooltip } from '@chakra-ui/react';
import _ from 'lodash';
import React, { ReactElement } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { RootState, useAppDispatch, useAppSelector } from '../../app/store';
import InpaintingWIP from '../../common/components/WorkInProgress/InpaintingWIP';
import NodesWIP from '../../common/components/WorkInProgress/NodesWIP';
import OutpaintingWIP from '../../common/components/WorkInProgress/OutpaintingWIP';
import { PostProcessingWIP } from '../../common/components/WorkInProgress/PostProcessingWIP';
import ImageToImageIcon from '../../common/icons/ImageToImageIcon';
import InpaintIcon from '../../common/icons/InpaintIcon';
import NodesIcon from '../../common/icons/NodesIcon';
import OutpaintIcon from '../../common/icons/OutpaintIcon';
import PostprocessingIcon from '../../common/icons/PostprocessingIcon';
import TextToImageIcon from '../../common/icons/TextToImageIcon';
import { setActiveTab } from '../options/optionsSlice';
import ImageToImage from './ImageToImage/ImageToImage';
import TextToImage from './TextToImage/TextToImage';

export const tab_dict = {
  txt2img: {
    title: <TextToImageIcon fill={'black'} boxSize={'2.5rem'} />,
    panel: <TextToImage />,
    tooltip: 'Text To Image',
  },
  img2img: {
    title: <ImageToImageIcon fill={'black'} boxSize={'2.5rem'} />,
    panel: <ImageToImage />,
    tooltip: 'Image To Image',
  },
  inpainting: {
    title: <InpaintIcon fill={'black'} boxSize={'2.5rem'} />,
    panel: <InpaintingWIP />,
    tooltip: 'Inpainting',
  },
  outpainting: {
    title: <OutpaintIcon fill={'black'} boxSize={'2.5rem'} />,
    panel: <OutpaintingWIP />,
    tooltip: 'Outpainting',
  },
  nodes: {
    title: <NodesIcon fill={'black'} boxSize={'2.5rem'} />,
    panel: <NodesWIP />,
    tooltip: 'Nodes',
  },
  postprocess: {
    title: <PostprocessingIcon fill={'black'} boxSize={'2.5rem'} />,
    panel: <PostProcessingWIP />,
    tooltip: 'Post Processing',
  },
};

export const tabMap = _.map(tab_dict, (tab, key) => key);

export default function InvokeTabs() {
  const activeTab = useAppSelector(
    (state: RootState) => state.options.activeTab
  );
  const dispatch = useAppDispatch();

  useHotkeys('1', () => {
    dispatch(setActiveTab(0));
  });

  useHotkeys('2', () => {
    dispatch(setActiveTab(1));
  });

  useHotkeys('3', () => {
    dispatch(setActiveTab(2));
  });

  useHotkeys('4', () => {
    dispatch(setActiveTab(3));
  });

  useHotkeys('5', () => {
    dispatch(setActiveTab(4));
  });

  useHotkeys('6', () => {
    dispatch(setActiveTab(5));
  });

  const renderTabs = () => {
    const tabsToRender: ReactElement[] = [];
    Object.keys(tab_dict).forEach((key) => {
      tabsToRender.push(
        <Tooltip
          key={key}
          hasArrow
          label={tab_dict[key as keyof typeof tab_dict].tooltip}
          placement={'right'}
        >
          <Tab>{tab_dict[key as keyof typeof tab_dict].title}</Tab>
        </Tooltip>
      );
    });
    return tabsToRender;
  };

  const renderTabPanels = () => {
    const tabPanelsToRender: ReactElement[] = [];
    Object.keys(tab_dict).forEach((key) => {
      tabPanelsToRender.push(
        <TabPanel className="app-tabs-panel" key={key}>
          {tab_dict[key as keyof typeof tab_dict].panel}
        </TabPanel>
      );
    });
    return tabPanelsToRender;
  };

  return (
    <Tabs
      isLazy
      className="app-tabs"
      variant={'unstyled'}
      defaultIndex={activeTab}
      index={activeTab}
      onChange={(index: number) => {
        dispatch(setActiveTab(index));
      }}
    >
      <div className="app-tabs-list">{renderTabs()}</div>
      <TabPanels className="app-tabs-panels">{renderTabPanels()}</TabPanels>
    </Tabs>
  );
}
