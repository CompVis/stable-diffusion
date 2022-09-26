import { extendTheme } from '@chakra-ui/react';
import type { StyleFunctionProps } from '@chakra-ui/styled-system';

export const theme = extendTheme({
  config: {
    initialColorMode: 'dark',
    useSystemColorMode: false,
  },
  components: {
    Tooltip: {
      baseStyle: (props: StyleFunctionProps) => ({
        textColor: props.colorMode === 'dark' ? 'gray.800' : 'gray.100',
      }),
    },
    Accordion: {
      baseStyle: (props: StyleFunctionProps) => ({
        button: {
          fontWeight: 'bold',
          _hover: {
            bgColor:
              props.colorMode === 'dark'
                ? 'rgba(255,255,255,0.05)'
                : 'rgba(0,0,0,0.05)',
          },
        },
        panel: {
          paddingBottom: 2,
        },
      }),
    },
    FormLabel: {
      baseStyle: {
        fontWeight: 'light',
      },
    },
    Button: {
      variants: {
        imageHoverIconButton: (props: StyleFunctionProps) => ({
          bg: props.colorMode === 'dark' ? 'blackAlpha.700' : 'whiteAlpha.800',
          color:
            props.colorMode === 'dark' ? 'whiteAlpha.700' : 'blackAlpha.700',
          _hover: {
            bg:
              props.colorMode === 'dark' ? 'blackAlpha.800' : 'whiteAlpha.800',
            color:
              props.colorMode === 'dark' ? 'whiteAlpha.900' : 'blackAlpha.900',
          },
        }),
      },
    },
  },
});
