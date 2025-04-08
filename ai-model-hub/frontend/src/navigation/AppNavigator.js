import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import { Ionicons } from '@expo/vector-icons';

// 導入屏幕（稍後實現）
import HomeScreen from '../screens/HomeScreen';
import ModelsScreen from '../screens/ModelsScreen';
import CompareScreen from '../screens/CompareScreen';
import SettingsScreen from '../screens/SettingsScreen';

const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();

// 主頁堆棧
const HomeStack = () => {
  return (
    <Stack.Navigator>
      <Stack.Screen name="Home" component={HomeScreen} options={{ headerShown: false }} />
      {/* 這裡可以添加更多與主頁相關的屏幕 */}
    </Stack.Navigator>
  );
};

// 模型堆棧
const ModelsStack = () => {
  return (
    <Stack.Navigator>
      <Stack.Screen name="Models" component={ModelsScreen} options={{ headerShown: false }} />
      {/* 這裡可以添加更多與模型相關的屏幕 */}
    </Stack.Navigator>
  );
};

// 比較堆棧
const CompareStack = () => {
  return (
    <Stack.Navigator>
      <Stack.Screen name="Compare" component={CompareScreen} options={{ headerShown: false }} />
      {/* 這裡可以添加更多與比較相關的屏幕 */}
    </Stack.Navigator>
  );
};

// 設置堆棧
const SettingsStack = () => {
  return (
    <Stack.Navigator>
      <Stack.Screen name="Settings" component={SettingsScreen} options={{ headerShown: false }} />
      {/* 這裡可以添加更多與設置相關的屏幕 */}
    </Stack.Navigator>
  );
};

// 底部標籤導航
const AppNavigator = () => {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ focused, color, size }) => {
            let iconName;

            if (route.name === 'HomeTab') {
              iconName = focused ? 'home' : 'home-outline';
            } else if (route.name === 'ModelsTab') {
              iconName = focused ? 'cube' : 'cube-outline';
            } else if (route.name === 'CompareTab') {
              iconName = focused ? 'git-compare' : 'git-compare-outline';
            } else if (route.name === 'SettingsTab') {
              iconName = focused ? 'settings' : 'settings-outline';
            }

            return <Ionicons name={iconName} size={size} color={color} />;
          },
        })}
        tabBarOptions={{
          activeTintColor: '#4a6ee0',
          inactiveTintColor: 'gray',
        }}
      >
        <Tab.Screen name="HomeTab" component={HomeStack} options={{ title: '首頁' }} />
        <Tab.Screen name="ModelsTab" component={ModelsStack} options={{ title: '模型' }} />
        <Tab.Screen name="CompareTab" component={CompareStack} options={{ title: '比較' }} />
        <Tab.Screen name="SettingsTab" component={SettingsStack} options={{ title: '設置' }} />
      </Tab.Navigator>
    </NavigationContainer>
  );
};

export default AppNavigator; 