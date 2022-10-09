import { IconButton, Link, Tooltip, useColorMode } from '@chakra-ui/react';
import { useHotkeys } from 'react-hotkeys-hook';

import { FaSun, FaMoon, FaGithub, FaDiscord } from 'react-icons/fa';
import { MdHelp, MdKeyboard, MdSettings } from 'react-icons/md';

import InvokeAILogo from '../../assets/images/logo.png';
import HotkeysModal from './HotkeysModal/HotkeysModal';

import SettingsModal from './SettingsModal/SettingsModal';
import StatusIndicator from './StatusIndicator';

/**
 * Header, includes color mode toggle, settings button, status message.
 */
const SiteHeader = () => {
  const { colorMode, toggleColorMode } = useColorMode();

  useHotkeys(
    'shift+d',
    () => {
      toggleColorMode();
    },
    [colorMode, toggleColorMode]
  );

  const colorModeIcon = colorMode == 'light' ? <FaMoon /> : <FaSun />;

  // Make FaMoon and FaSun icon apparent size consistent
  const colorModeIconFontSize = colorMode == 'light' ? 18 : 20;

  return (
    <div className="site-header">
      <div className="site-header-left-side">
        <img src={InvokeAILogo} alt="invoke-ai-logo" />
        <h1>
          invoke <strong>ai</strong>
        </h1>
      </div>

      <div className="site-header-right-side">
        <StatusIndicator />

        <SettingsModal>
          <IconButton
            aria-label="Settings"
            variant="link"
            fontSize={24}
            size={'sm'}
            icon={<MdSettings />}
          />
        </SettingsModal>

        <HotkeysModal>
          <IconButton
            aria-label="Hotkeys"
            variant="link"
            fontSize={24}
            size={'sm'}
            icon={<MdKeyboard />}
          />
        </HotkeysModal>

        <Tooltip hasArrow label="Report Bug" placement={'bottom'}>
          <IconButton
            aria-label="Link to Github Issues"
            variant="link"
            fontSize={23}
            size={'sm'}
            icon={
              <Link
                isExternal
                href="http://github.com/invoke-ai/InvokeAI/issues"
              >
                <MdHelp />
              </Link>
            }
          />
        </Tooltip>

        <Tooltip hasArrow label="Github" placement={'bottom'}>
          <IconButton
            aria-label="Link to Github Repo"
            variant="link"
            fontSize={20}
            size={'sm'}
            icon={
              <Link isExternal href="http://github.com/invoke-ai/InvokeAI">
                <FaGithub />
              </Link>
            }
          />
        </Tooltip>

        <Tooltip hasArrow label="Discord" placement={'bottom'}>
          <IconButton
            aria-label="Link to Discord Server"
            variant="link"
            fontSize={20}
            size={'sm'}
            icon={
              <Link isExternal href="https://discord.gg/ZmtBAhwWhy">
                <FaDiscord />
              </Link>
            }
          />
        </Tooltip>

        <Tooltip hasArrow label="Theme" placement={'bottom'}>
          <IconButton
            aria-label="Toggle Dark Mode"
            onClick={toggleColorMode}
            variant="link"
            size={'sm'}
            fontSize={colorModeIconFontSize}
            icon={colorModeIcon}
          />
        </Tooltip>
      </div>
    </div>
  );
};

export default SiteHeader;
