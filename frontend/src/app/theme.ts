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
  },
});
